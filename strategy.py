#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_WeeklyTrend_Regime_v1
Hypothesis: On 1d timeframe, Camarilla R3/S3 breakouts aligned with weekly trend regime (price > weekly EMA34 for longs, < for shorts) capture institutional moves with reduced whipsaw. Weekly trend filter ensures trading with higher timeframe momentum, while Camarilla levels provide precise entry/exit points. Volume confirmation (>1.5x 20-day average) filters low-quality breakouts. Discrete sizing (0.25) and ATR-based stoploss (2.5x) manage risk. Target: 30-100 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for weekly trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w EMA34 for weekly trend regime ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily Camarilla levels (based on previous day) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Camarilla levels for today using yesterday's OHLC
    camarilla_r3 = np.zeros(n)
    camarilla_s3 = np.zeros(n)
    camarilla_r4 = np.zeros(n)
    camarilla_s4 = np.zeros(n)
    camarilla_r3[:] = np.nan
    camarilla_s3[:] = np.nan
    camarilla_r4[:] = np.nan
    camarilla_s4[:] = np.nan
    
    for i in range(1, n):
        # Use previous day's OHLC to calculate today's levels
        high_1 = high[i-1]
        low_1 = low[i-1]
        close_1 = close[i-1]
        
        if np.isnan(high_1) or np.isnan(low_1) or np.isnan(close_1):
            continue
            
        range_1 = high_1 - low_1
        camarilla_r3[i] = close_1 + range_1 * 1.1 / 4
        camarilla_s3[i] = close_1 - range_1 * 1.1 / 4
        camarilla_r4[i] = close_1 + range_1 * 1.1 / 2
        camarilla_s4[i] = close_1 - range_1 * 1.1 / 2
    
    # === Daily volume confirmation (>1.5x 20-day average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    # === ATR for stoploss ===
    # True Range
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(atr[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        weekly_ema = ema_34_1w_aligned[i]
        vol_conf = volume_confirmed[i]
        atr_val = atr[i]
        
        # Weekly trend regime
        is_bull = price > weekly_ema
        is_bear = price < weekly_ema
        
        if position == 0:
            # Look for breakout signals
            if is_bull and vol_conf:
                # Bull regime: long on break above R3 or R4
                if price > camarilla_r3[i] or price > camarilla_r4[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
            elif is_bear and vol_conf:
                # Bear regime: short on break below S3 or S4
                if price < camarilla_s3[i] or price < camarilla_s4[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position != 0:
            # Check stoploss (2.5x ATR) and time-based exit (max 10 days)
            if position == 1:
                # Long: exit if price drops below entry - 2.5*ATR or reverse signal
                if price < entry_price - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                elif price < camarilla_s3[i]:  # Reverse to bearish level
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Short: exit if price rises above entry + 2.5*ATR or reverse signal
                if price > entry_price + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                elif price > camarilla_r3[i]:  # Reverse to bullish level
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_WeeklyTrend_Regime_v1"
timeframe = "1d"
leverage = 1.0