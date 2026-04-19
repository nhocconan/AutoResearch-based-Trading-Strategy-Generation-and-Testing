#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour breakout from daily Donchian channels with volume confirmation and ADX trend filter.
# Uses daily Donchian(20) breakouts as the primary signal, filtered by ADX(14) > 25 to ensure
# trending markets and volume spike > 1.5x 20-period average to confirm breakout strength.
# Designed for 6h timeframe to capture medium-term trends with controlled frequency (~15-25 trades/year).
# Exit on opposite Donchian touch or ADX weakening (< 20) to avoid whipsaws in ranging markets.
name = "6h_DailyDonchian20_ADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on daily data
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align to 6h timeframe (waits for prior day close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # ADX(14) for trend strength
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/14)
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    adx = np.zeros(n)
    
    atr[13] = np.mean(tr[1:14]) if n > 13 else 0
    plus_dm_smoothed = np.zeros(n)
    minus_dm_smoothed = np.zeros(n)
    
    if n > 13:
        plus_dm_smoothed[13] = np.sum(plus_dm[1:14])
        minus_dm_smoothed[13] = np.sum(minus_dm[1:14])
    
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        plus_dm_smoothed[i] = (plus_dm_smoothed[i-1] * 13 + plus_dm[i]) / 14
        minus_dm_smoothed[i] = (minus_dm_smoothed[i-1] * 13 + minus_dm[i]) / 14
        
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm_smoothed[i] / atr[i]
            minus_di[i] = 100 * minus_dm_smoothed[i] / atr[i]
            dx = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) > 0 else 0
        else:
            plus_di[i] = 0
            minus_di[i] = 0
            dx = 0
        
        if i >= 27:  # Need 14 periods of DX for ADX
            if i == 27:
                adx[i] = np.mean(dx[14:28])
            else:
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Volume spike: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high with strong trend and volume
            if (close[i] > donchian_high_aligned[i] and 
                adx[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with strong trend and volume
            elif (close[i] < donchian_low_aligned[i] and 
                  adx[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches Donchian low or trend weakens
            if (close[i] < donchian_low_aligned[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches Donchian high or trend weakens
            if (close[i] > donchian_high_aligned[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals