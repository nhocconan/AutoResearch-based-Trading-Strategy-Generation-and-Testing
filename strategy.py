#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using Williams Alligator (13,8,5 SMAs) + 1w EMA50 trend filter + volume confirmation
# Williams Alligator identifies trend via jaw-teeth-lips alignment: bullish when lips>teeth>jaw, bearish when lips<teeth<jaw.
# 1w EMA50 filters for weekly trend alignment to avoid counter-trend trades.
# Volume confirmation (2.0x 20-period average) ensures institutional participation.
# Uses 1d timeframe for lower trade frequency (target: 30-100 trades over 4 years) to minimize fee drag.
# Discrete sizing 0.25 to balance risk and return. Session filter (08-20 UTC) reduces noise.

name = "1d_Williams_Alligator_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator: SMAs of median price (HL/2)
    # Jaw: 13-period SMA, shifted 8 bars
    # Teeth: 8-period SMA, shifted 5 bars
    # Lips: 5-period SMA, shifted 3 bars
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Williams Alligator to 1d timeframe (already aligned via get_htf_data)
    jaw_aligned = jaw  # get_htf_data('1d') returns 1d data aligned to 1d prices
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_lips = lips_aligned[i]
        curr_teeth = teeth_aligned[i]
        curr_jaw = jaw_aligned[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: lips > teeth > jaw (bullish alignment) AND price > 1w EMA50 (uptrend)
                if curr_lips > curr_teeth and curr_teeth > curr_jaw and curr_close > curr_ema_50_1w:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: lips < teeth < jaw (bearish alignment) AND price < 1w EMA50 (downtrend)
                elif curr_lips < curr_teeth and curr_teeth < curr_jaw and curr_close < curr_ema_50_1w:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Alligator alignment breaks bearish OR price falls below 1w EMA50
            if curr_lips < curr_teeth or curr_teeth < curr_jaw or curr_close < curr_ema_50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Alligator alignment breaks bullish OR price rises above 1w EMA50
            if curr_lips > curr_teeth or curr_teeth > curr_jaw or curr_close > curr_ema_50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals