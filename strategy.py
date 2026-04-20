#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA(50) trend filter and volume confirmation
# - Long when price breaks above upper Donchian(20) channel AND 1w EMA(50) > 1w EMA(50) 1 period ago (uptrend)
# - Short when price breaks below lower Donchian(20) channel AND 1w EMA(50) < 1w EMA(50) 1 period ago (downtrend)
# - Volume confirmation: current volume > 1.5 * average volume of last 20 periods
# - Stop loss: exit when price crosses the opposite Donchian band
# - Designed for 12h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1w data for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w timeframe
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels on 12h timeframe
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    upper_channel = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after Donchian and EMA warmup
        # Skip if NaN in indicators
        if np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        volume = volume_12h[i]
        upper = upper_channel[i]
        lower = lower_channel[i]
        ema = ema_50_1w_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above upper channel AND uptrend (EMA rising) AND volume confirmation
            if price > upper and ema > ema_50_1w_aligned[i-1] and volume > 1.5 * avg_volume[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower channel AND downtrend (EMA falling) AND volume confirmation
            elif price < lower and ema < ema_50_1w_aligned[i-1] and volume > 1.5 * avg_volume[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower channel (opposite band)
            if price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper channel (opposite band)
            if price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_1wEMA50_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0