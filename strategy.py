#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper band AND price > 1w EMA50 AND volume > 1.8x 20-bar avg
# Short when price breaks below Donchian lower band AND price < 1w EMA50 AND volume > 1.8x 20-bar avg
# Exit when price crosses Donchian median (mean reversion to center)
# Uses discrete position sizing (0.25) to minimize fee drag while maintaining sufficient exposure.
# Target: 15-30 trades/year on 1d (60-120 total over 4 years).
# Donchian channels provide clear structure; 1w EMA50 filters counter-trend moves in bear markets.
# Volume spike ensures institutional participation, reducing false breakouts.

name = "1d_Donchian20_1wEMA50_VolumeBreakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on 1w data
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) from price data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_median = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # EMA50 and Donchian warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(donchian_median[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_1w_aligned[i]
        
        # Donchian levels
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        median = donchian_median[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below Donchian median (mean reversion to center)
            if curr_close < median:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian median (mean reversion to center)
            if curr_close > median:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above upper band AND price > 1w EMA50 AND volume confirmation
            if curr_close > upper and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower band AND price < 1w EMA50 AND volume confirmation
            elif curr_close < lower and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals