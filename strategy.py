#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel with 1w uptrend (price > 1w EMA50) and volume spike.
# Short when price breaks below lower Donchian channel with 1w downtrend (price < 1w EMA50) and volume spike.
# Exit when price returns to the midpoint of the Donchian channel (mean reversion).
# Donchian channels provide structural support/resistance; 1w EMA50 filters trend; volume confirms participation.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirmation_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels from previous 20-period (20 days)
    # Need 20-period high and low for Donchian calculation
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels (they are already calculated on 1d timeframe, so no need for HTF alignment)
    # But we need to shift by 1 to use previous completed bar for breakout
    upper_channel = np.roll(high_ma_20, 1)  # previous bar's 20-period high
    lower_channel = np.roll(low_ma_20, 1)   # previous bar's 20-period low
    # Set first value to nan to avoid look-ahead
    upper_channel[0] = np.nan
    lower_channel[0] = np.nan
    
    # Calculate midpoint for exit
    midpoint = (high_ma_20 + low_ma_20) / 2.0
    midpoint_aligned = np.roll(midpoint, 1)  # previous bar's midpoint
    midpoint_aligned[0] = np.nan
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(midpoint_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_upper = upper_channel[i]
        curr_lower = lower_channel[i]
        curr_mid = midpoint_aligned[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper channel, uptrend (price > 1w EMA50), volume confirmation
            if (curr_close > curr_upper and 
                curr_close > curr_ema_50_1w and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel, downtrend (price < 1w EMA50), volume confirmation
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_50_1w and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price returns to midpoint (mean reversion)
            if curr_close <= curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price returns to midpoint (mean reversion)
            if curr_close >= curr_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals