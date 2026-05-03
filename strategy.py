#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
# Uses ATR-based trailing stop for risk management. Discrete sizing 0.25.
# Target: 30-100 total trades over 4 years (7-25/year).
# Donchian channels provide clear breakout levels that work in both trending and ranging markets.
# 1w EMA34 filter ensures we only trade in the direction of the higher timeframe trend.
# Volume confirmation reduces false breakouts.
# This strategy focuses on BTC and ETH as primary targets, avoiding SOL-only bias.

name = "1d_Donchian20_1wEMA34_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w Donchian(20) levels (upper, lower) from prior completed 1w bar
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:  # Need at least 1 completed bar for prior
        return np.zeros(n)
    
    # Calculate prior completed 1w bar's high, low, close for Donchian
    prior_high_1w = np.roll(df_1w['high'].values, 1)
    prior_low_1w = np.roll(df_1w['low'].values, 1)
    prior_high_1w[0] = np.nan
    prior_low_1w[0] = np.nan
    
    # Calculate 20-period Donchian channels from prior data
    high_series = pd.Series(prior_high_1w)
    low_series = pd.Series(prior_low_1w)
    donchian_upper_1w = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower_1w = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_1w)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_1w)
    
    # Calculate 1w EMA34 trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR(24) for stoploss (using 1d data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=24, min_periods=24, adjust=False).mean().values
    
    # Volume confirmation: volume > 2.0x 24-bar average (on 1d data)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(100, n):  # Start after sufficient warmup
        # Get current values
        upper_val = donchian_upper_aligned[i]
        lower_val = donchian_lower_aligned[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(upper_val) or np.isnan(lower_val) or np.isnan(ema_trend) or np.isnan(atr_val):
            continue
            
        # Entry conditions
        # Long: break above Donchian upper with volume spike and above 1w EMA34
        long_entry = (close[i] > upper_val) and (close[i] > ema_trend) and vol_spike
        # Short: break below Donchian lower with volume spike and below 1w EMA34
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