# 4H_1D_VOLUME_SPIKE_BREAKOUT
# Hypothesis: In both bull and bear markets, strong volume spikes during breakouts from key daily levels (support/resistance) indicate institutional participation and higher probability of follow-through. Using 1-day high/low as dynamic support/resistance with volume confirmation on 4H timeframe reduces false breakouts. The strategy works in trending markets (breakouts continue) and ranging markets (mean reversion at extremes with volume confirmation).
# Timeframe: 4H balances trade frequency and signal quality, avoiding excessive fees while capturing significant moves.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR for volatility filter (14-period)
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align daily ATR to 4H timeframe
    atr_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate volume moving average (20-period) for 4H
    vol_ma_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(200, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_4h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.3% of price)
        if atr_4h[i] < 0.003 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        # Get previous day's high and low for support/resistance
        prev_high = high_1d[i-1] if i > 0 else high_1d[0]
        prev_low = low_1d[i-1] if i > 0 else low_1d[0]
        
        if position == 0:
            # Long: Price breaks above previous day's high with volume confirmation
            if close[i] > prev_high and volume_ratio > vol_threshold:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below previous day's low with volume confirmation
            elif close[i] < prev_low and volume_ratio > vol_threshold:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below previous day's low
            if close[i] < prev_low:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above previous day's high
            if close[i] > prev_high:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4H_1D_Volume_Spike_Breakout"
timeframe = "4h"
leverage = 1.0