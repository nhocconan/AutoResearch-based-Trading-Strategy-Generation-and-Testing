#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and ATR-based volume confirmation.
# Long when price breaks above upper Donchian channel, price > 12h EMA50, and volume > 1.5x ATR-scaled average volume.
# Short when price breaks below lower Donchian channel, price < 12h EMA50, and volume > 1.5x ATR-scaled average volume.
# Exit when price reverts to the middle of the Donchian channel (mean reversion).
# Uses 12h EMA50 for higher timeframe trend alignment and ATR-scaled volume to avoid low-momentum breakouts.
# Targets 20-50 trades/year on 4h timeframe with discrete position sizing (0.25) to minimize fee drag.
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend alignment.

name = "4h_Donchian20_12hEMA50_ATRVolConfirm_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) for volume confirmation scaling
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR-scaled average volume (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    atr_scaled_vol_ma = vol_ma_20 * (atr / np.mean(atr))  # normalize by average ATR
    volume_confirm = volume > (1.5 * atr_scaled_vol_ma)
    
    # Calculate Donchian channels (20-period)
    upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_channel = (upper_channel + lower_channel) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for EMA50, ATR, and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(middle_channel[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_upper = upper_channel[i]
        curr_lower = lower_channel[i]
        curr_middle = middle_channel[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian, price > 12h EMA50, volume confirmation
            if (curr_close > curr_upper and 
                curr_close > curr_ema_50_12h and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, price < 12h EMA50, volume confirmation
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_50_12h and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price reverts to middle of Donchian channel (mean reversion)
            if curr_close <= curr_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price reverts to middle of Donchian channel (mean reversion)
            if curr_close >= curr_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals