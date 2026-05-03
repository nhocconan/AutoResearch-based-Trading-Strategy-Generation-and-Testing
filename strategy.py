#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above 20-period high with volume > 1.5x 20-period MA and price > 1d EMA50.
# Short when price breaks below 20-period low with volume spike and price < 1d EMA50.
# Exit: ATR(14) trailing stop (2.0x ATR) or Donchian opposite break.
# Discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Donchian provides objective breakout levels; 1d EMA50 filters counter-trend trades in bear markets;
# volume confirmation reduces false breakouts. Works in bull via trend-following breakouts
# and in bear via short breakdowns with trend alignment.

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
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[tr[0]], tr])  # same length as prices
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 4h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long ATR stop
    lowest_since_entry = 0.0   # for short ATR stop
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        upper_donchian = highest_20[i]
        lower_donchian = lowest_20[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Update highest/lowest since entry for ATR stop
        if position == 1:
            highest_since_entry = max(highest_since_entry, high[i])
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, low[i]) if lowest_since_entry != 0 else low[i]
        
        # Entry logic
        if position == 0:
            # Long: break above upper Donchian with volume spike in uptrend
            if close_val > upper_donchian and vol_spike and is_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = high[i]
            # Short: break below lower Donchian with volume spike in downtrend
            elif close_val < lower_donchian and vol_spike and is_downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = low[i]
        elif position == 1:
            # Long exit: ATR stoploss OR price breaks below lower Donchian OR trend turns down
            atr_stop = highest_since_entry - (2.0 * atr_val)
            if close_val < atr_stop or close_val < lower_donchian or not is_uptrend:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ATR stoploss OR price breaks above upper Donchian OR trend turns up
            atr_stop = lowest_since_entry + (2.0 * atr_val)
            if close_val > atr_stop or close_val > upper_donchian or not is_downtrend:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals