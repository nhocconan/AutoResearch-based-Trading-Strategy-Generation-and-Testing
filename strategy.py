#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA200 trend filter + volume confirmation
# - Long when price breaks above 20-day high AND price > 1w EMA200 (uptrend) AND volume > 1.5x 20-day volume SMA
# - Short when price breaks below 20-day low AND price < 1w EMA200 (downtrend) AND volume > 1.5x 20-day volume SMA
# - Exit: opposing Donchian breakout or volume drops below average
# - Uses 1d for price action and volume, 1w for trend filter
# - Target: 15-25 trades/year to minimize fee drag while capturing strong trending moves
# - Donchian breakouts work in both bull and bear markets when filtered by higher timeframe trend
# - Volume confirmation reduces false breakouts

name = "1d_1w_donchian_trend_volume_v1"
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
    
    # Load 1w data ONCE before loop for trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Calculate 20-day Donchian channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Pre-compute 1d volume for alignment
    volume_aligned = align_htf_to_ltf(prices, df_1w, volume)  # Using 1w index for volume alignment
    
    for i in range(200, n):  # Start after warmup for EMA200
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(volume_sma_20[i]) or np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Only trade when volume confirmation is present
        if vol_confirm:
            # Long: price breaks above 20-day high AND price > 1w EMA200 (uptrend)
            if close[i] > highest_high_20[i] and close[i] > ema_200_1w_aligned[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: price breaks below 20-day low AND price < 1w EMA200 (downtrend)
            elif close[i] < lowest_low_20[i] and close[i] < ema_200_1w_aligned[i]:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            
            # Exit conditions: opposing Donchian breakout
            if position == 1 and close[i] < lowest_low_20[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > highest_high_20[i]:
                position = 0
                signals[i] = 0.0
        else:
            # No volume confirmation: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals