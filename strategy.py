#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w HMA trend filter and volume confirmation
# - Enter long when price breaks above 20-day Donchian upper band AND 1w HMA(21) is rising AND 1d volume > 1.5x 20-day volume SMA
# - Enter short when price breaks below 20-day Donchian lower band AND 1w HMA(21) is falling AND 1d volume > 1.5x 20-day volume SMA
# - Exit: time-based exit after 10 bars or opposite Donchian break
# - Donchian channels provide clear structural breakouts
# - 1w HMA(21) filter ensures alignment with higher timeframe trend to avoid counter-trend trades
# - Volume confirmation increases breakout validity
# - Target: 15-25 trades/year to minimize fee drag while capturing high-probability breakouts

name = "1d_1w_donchian_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop for HMA trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute 20-period Donchian channels for 1d
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 20-period volume SMA for 1d
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute HMA for 1w close (trend filter)
    close_1w = df_1w['close'].values
    # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = len(close_1w) // 2
    sqrt_len = int(np.sqrt(len(close_1w)))
    if half_len > 0 and sqrt_len > 0:
        wma_half = pd.Series(close_1w).ewm(span=half_len, adjust=False, min_periods=half_len).mean()
        wma_full = pd.Series(close_1w).ewm(span=len(close_1w), adjust=False, min_periods=len(close_1w)).mean()
        raw_hma = 2 * wma_half - wma_full
        hma_21 = raw_hma.ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    else:
        hma_21 = np.full_like(close_1w, np.nan)
    
    # Align 1w HMA to 1d timeframe (using completed 1w bar)
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # Pre-compute HMA slope for trend direction (rising/falling)
    hma_slope = np.diff(hma_21_aligned, prepend=hma_21_aligned[0])
    hma_rising = hma_slope > 0
    hma_falling = hma_slope < 0
    
    for i in range(30, n):  # Start after 30-bar warmup for 20-period indicators
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(hma_21_aligned[i]) or
            np.isnan(hma_rising[i]) or np.isnan(hma_falling[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Donchian breakout signals
        long_breakout = close[i] > highest_high_20[i]
        short_breakout = close[i] < lowest_low_20[i]
        
        # Exit conditions: time-based or opposite breakout
        # We'll use a simple approach: exit on opposite Donchian break or after 10 bars
        # Track bars in position separately
        
        # Trading logic
        if long_breakout and vol_confirm and hma_rising[i]:
            if position != 1:  # Only signal on new long entry
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif short_breakout and vol_confirm and hma_falling[i]:
            if position != -1:  # Only signal on new short entry
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = -0.25
        else:
            # Exit conditions: opposite Donchian break
            if position == 1 and short_breakout:
                position = 0
                signals[i] = 0.0
            elif position == -1 and long_breakout:
                position = 0
                signals[i] = 0.0
            else:
                # Maintain current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals