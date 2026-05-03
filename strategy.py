#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based stoploss.
# Uses Donchian channel breakouts for entry, aligned with 1d EMA50 trend direction.
# Long when price breaks above 20-period high with volume > 1.3x 20-period MA and close > 1d EMA50.
# Short when price breaks below 20-period low with volume spike and close < 1d EMA50.
# ATR stoploss exits when price moves against position by 2.5x ATR(14).
# Discrete sizing 0.25. Target: 80-180 total trades over 4 years (20-45/year).
# Donchian provides clear structure, EMA50 filters counter-trend trades, volume confirmation reduces false breakouts.
# Works in bull via trend-following breaks and in bear via short breakdowns with trend alignment.

name = "4h_Donchian20_1dEMA50_Volume_ATR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 4h volume > 1.3x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: break above upper Donchian with volume spike in uptrend
            if close_val > upper and vol_spike and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume spike in downtrend
            elif close_val < lower and vol_spike and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below lower Donchian OR ATR stoploss hit
            if close_val < lower or close_val < (entry_price - 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper Donchian OR ATR stoploss hit
            if close_val > upper or close_val > (entry_price + 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        
        # Track entry price for stoploss calculation
        if position != 0 and signals[i] != 0.0 and position != 0:
            if i == 50 or (position == 1 and signals[i-1] == 0.0) or (position == -1 and signals[i-1] == 0.0):
                entry_price = close_val
    
    return signals