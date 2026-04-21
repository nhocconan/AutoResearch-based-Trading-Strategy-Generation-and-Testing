#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_v2
Hypothesis: Camarilla pivot levels (R1, S1) from 1d timeframe provide intraday support/resistance. Breakout above R1 or below S1 with 1d EMA34 trend filter and volume confirmation (volume > 1.5x 20-period average) captures institutional flow. ATR-based stoploss limits downside. Designed for 4h timeframe to work in both bull (breakouts) and bear (mean reversion at extremes) markets. Discrete sizing (0.25) minimizes fee churn. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla and EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # === 1d Camarilla pivot levels (based on previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for today based on yesterday's OHLC
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    rng = high_1d - low_1d
    r1 = close_1d + rng * 1.1 / 12
    s1 = close_1d - rng * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1d EMA34 for trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 4h volume confirmation (volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    # === 4h ATR for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_34_val = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: breakout above R1 + uptrend (price > EMA34) + volume spike
            if price > r1_val and price > ema_34_val and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: breakdown below S1 + downtrend (price < EMA34) + volume spike
            elif price < s1_val and price < ema_34_val and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # ATR-based stoploss and profit taking
            if position == 1:
                # Stoploss: 2 * ATR below entry
                if price < entry_price - 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                # Take profit: exit when price re-tests Camarilla levels
                elif price < r1_val:  # price breaks below R1
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Stoploss: 2 * ATR above entry
                if price > entry_price + 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                # Take profit: exit when price re-tests Camarilla levels
                elif price > s1_val:  # price breaks above S1
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_v2"
timeframe = "4h"
leverage = 1.0