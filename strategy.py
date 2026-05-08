#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (13,8,5 SMAs) with 1w trend filter and volume confirmation.
# Long when Jaw > Teeth > Lips (bullish alignment) AND price > 1w EMA50 AND volume > 1.5x 20-period avg.
# Short when Jaw < Teeth < Lips (bearish alignment) AND price < 1w EMA50 AND volume > 1.5x 20-period avg.
# Exit when Alligator alignment breaks or price crosses 1w EMA50.
# Williams Alligator identifies trend phases; 1w EMA50 filters higher-timeframe trend; volume confirms conviction.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.

name = "12h_WilliamsAlligator_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Williams Alligator on 12h data: SMAs of median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # Blue line (13-bar)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # Red line (8-bar)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values    # Green line (5-bar)
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # EMA50 on 1w close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator alignment
        bullish_alignment = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        bearish_alignment = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        
        if position == 0:
            # Long conditions: bullish alignment, price > 1w EMA50, volume filter
            long_cond = bullish_alignment and (close[i] > ema_50_1w_aligned[i]) and volume_filter[i]
            # Short conditions: bearish alignment, price < 1w EMA50, volume filter
            short_cond = bearish_alignment and (close[i] < ema_50_1w_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: alignment breaks or price < 1w EMA50
            if not bullish_alignment or (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: alignment breaks or price > 1w EMA50
            if not bearish_alignment or (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals