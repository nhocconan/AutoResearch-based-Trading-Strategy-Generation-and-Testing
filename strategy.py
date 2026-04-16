#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) extreme reversal with 1d volume confirmation and 1w trend filter.
# Williams %R < -80 = oversold, > -20 = overbought.
# Long when Williams %R crosses above -80 from below AND 1d volume > 1.5x 20-period average AND 1w close > 1w EMA(50) (bullish weekly trend).
# Short when Williams %R crosses below -20 from above AND 1d volume > 1.5x 20-period average AND 1w close < 1w EMA(50) (bearish weekly trend).
# Exit when Williams %R crosses opposite extreme (-20 for long, -80 for short) or ATR-based stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture mean reversals in alignment with weekly trend.
# Works in both bull and bear markets by using weekly trend filter and volume confirmation, avoiding counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Williams %R (14) ===
    highest_high_6h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_6h - close) / (highest_high_6h - lowest_low_6h)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 1w Indicators: EMA(50) for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    weekly_uptrend = close_1w > ema_50_1w_aligned  # bullish weekly trend
    weekly_downtrend = close_1w < ema_50_1w_aligned  # bearish weekly trend
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for weekly EMA)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Calculate 6h ATR for stoploss
    tr1_6h = pd.Series(high).diff()
    tr2_6h = pd.Series(low).diff().abs()
    tr3_6h = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1_6h, tr2_6h, tr3_6h], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(williams_r[i]) or np.isnan(volume_spike[i]) or np.isnan(weekly_uptrend[i]) or
            np.isnan(weekly_downtrend[i]) or np.isnan(atr_6h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_weekly_uptrend = weekly_uptrend[i]
        is_weekly_downtrend = weekly_downtrend[i]
        atr_val = atr_6h_raw[i]
        
        # Williams %R extremes
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R crosses above -20 (overbought)
            if wr > -20 and wr_prev <= -20:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R crosses below -80 (oversold)
            if wr < -80 and wr_prev >= -80:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R crosses above -80 from below AND volume spike AND weekly uptrend
            if wr > -80 and wr_prev <= -80 and vol_spike and is_weekly_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Williams %R crosses below -20 from above AND volume spike AND weekly downtrend
            elif wr < -20 and wr_prev >= -20 and vol_spike and is_weekly_downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_WilliamsR_1dVolumeSpike_1wEMA50_V1"
timeframe = "6h"
leverage = 1.0