#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Uses ATR-based trailing stop for risk management. Discrete sizing 0.25.
# Target: 50-150 total trades over 4 years (12-37/year).
# Donchian breakouts capture strong momentum moves that work in both bull and bear markets.
# 1d EMA50 filter ensures we only trade in the direction of the higher timeframe trend.
# Volume confirmation reduces false breakouts.

name = "12h_Donchian20_1dEMA50_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian(20) channels from prior completed 1d bar
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 1 completed bar for prior
        return np.zeros(n)
    
    # Calculate upper and lower channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Rolling max/min for Donchian channels
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only prior completed bar (avoid look-ahead)
    upper_20_prior = np.roll(upper_20, 1)
    lower_20_prior = np.roll(lower_20, 1)
    upper_20_prior[0] = np.nan
    lower_20_prior[0] = np.nan
    
    # Align Donchian levels to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_20_prior)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_20_prior)
    
    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(24) for stoploss (using 12h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=24, min_periods=24, adjust=False).mean().values
    
    # Volume confirmation: volume > 2.0x 24-bar average (on 12h data)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(100, n):  # Start after sufficient warmup
        # Get current values
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(upper_val) or np.isnan(lower_val) or np.isnan(ema_trend) or np.isnan(atr_val):
            continue
            
        # Entry conditions
        # Long: break above upper Donchian with volume spike and above 1d EMA50
        long_entry = (close[i] > upper_val) and (close[i] > ema_trend) and vol_spike
        # Short: break below lower Donchian with volume spike and below 1d EMA50
        short_entry = (close[i] < lower_val) and (close[i] < ema_trend) and vol_spike
        
        # Exit conditions (trailing stop)
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            long_exit = close[i] < (highest_high_since_entry - 2.5 * atr_val)
        elif position == -1:  # Short position
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            short_exit = close[i] > (lowest_low_since_entry + 2.5 * atr_val)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals