#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation.
# Long when price breaks above Donchian(20) high AND price > 1d EMA34 AND volume > 1.5x 20-bar average.
# Short when price breaks below Donchian(20) low AND price < 1d EMA34 AND volume > 1.5x 20-bar average.
# Uses discrete position sizing (0.25) to limit drawdown and fee churn. Works in bull/bear via 1d EMA34 trend filter.
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_Donchian20_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34, 20)  # warmup for EMA34, Donchian, and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above Donchian high, uptrend, volume confirmation
            if (curr_high > donchian_high[i] and 
                curr_close > ema_34_1d_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low, downtrend, volume confirmation
            elif (curr_low < donchian_low[i] and 
                  curr_close < ema_34_1d_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit when price breaks below Donchian low or trend changes
            if curr_low < donchian_low[i] or curr_close < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high or trend changes
            if curr_high > donchian_high[i] or curr_close > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals