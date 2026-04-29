#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (>1.5x average)
# Donchian channels provide clear structure for breakout trading; 1d EMA50 ensures alignment with higher timeframe trend
# Volume confirmation filters out false breakouts; discrete position sizing (0.25) minimizes fee churn
# Designed for 12-37 trades/year on 12h timeframe to avoid overtrading while capturing momentum moves
# Works in both bull and bear markets via 1d EMA50 trend filter - only trades in direction of higher timeframe momentum

name = "12h_Donchian20_1dEMA50_VolumeBreakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) for breakout levels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # Donchian and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low (failed breakout) or reversal signal
            if curr_close < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high (failed breakout) or reversal signal
            if curr_close > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new breakout entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long breakout when price closes above Donchian high with 1d EMA50 uptrend and volume confirmation
            if curr_close > donchian_high[i] and curr_close > curr_ema50_1d and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short breakout when price closes below Donchian low with 1d EMA50 downtrend and volume confirmation
            elif curr_close < donchian_low[i] and curr_close < curr_ema50_1d and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals