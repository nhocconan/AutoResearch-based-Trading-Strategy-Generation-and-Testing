#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w EMA50 trend filter and 1d volume confirmation.
# Long when price breaks above 20-bar Donchian high, price > 1w EMA50, and 1d volume > 1.5x 20-bar avg.
# Short when price breaks below 20-bar Donchian low, price < 1w EMA50, and 1d volume > 1.5x 20-bar avg.
# Exit when price crosses the 10-bar EMA (mean reversion on 6h).
# Uses 1w EMA50 for higher timeframe trend alignment, targeting 12-37 trades/year on 6h.
# Trend filter avoids counter-trend trades, volume confirmation reduces false signals.
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend alignment.

name = "6h_Donchian20_1wEMA50_1dVolume_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d volume 20-period average for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h EMA10 for exit signal
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for Donchian and 1w EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_10[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_ema_10 = ema_10[i]
        curr_vol_1d = volume_1d[i] if i < len(volume_1d) else volume_1d[-1]  # fallback for alignment
        curr_vol_ma_20_1d = vol_ma_20_1d_aligned[i]
        
        # Volume confirmation: 1d volume > 1.5x 20-period average
        volume_confirm = curr_vol_1d > (1.5 * curr_vol_ma_20_1d)
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high, price > 1w EMA50, volume spike
            if (curr_close > curr_donchian_high and 
                curr_close > curr_ema_50_1w and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, price < 1w EMA50, volume spike
            elif (curr_close < curr_donchian_low and 
                  curr_close < curr_ema_50_1w and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below 10-bar EMA (mean reversion)
            if curr_close < curr_ema_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above 10-bar EMA (mean reversion)
            if curr_close > curr_ema_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals