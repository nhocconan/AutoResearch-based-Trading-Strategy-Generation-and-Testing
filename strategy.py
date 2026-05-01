#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above upper Donchian(20) AND 1w EMA50 uptrend AND volume > 1.5x 20-period median.
# Short when price breaks below lower Donchian(20) AND 1w EMA50 downtrend AND volume > 1.5x 20-period median.
# Donchian channels capture volatility expansion and structure; 1w EMA50 filters for higher-timeframe trend alignment; volume spike confirms breakout conviction.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
# Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years) to minimize fee drag.

name = "1d_Donchian20_Breakout_1wEMA50_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian Channels (20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    dc_upper = high_series.rolling(window=20, min_periods=20).max().values
    dc_lower = low_series.rolling(window=20, min_periods=20).min().values
    dc_middle = (dc_upper + dc_lower) / 2.0
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate 1w EMA50 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for Donchian, EMA, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1w EMA50 direction
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper Donchian AND uptrend AND volume spike
            if curr_close > dc_upper[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below lower Donchian AND downtrend AND volume spike
            elif curr_close < dc_lower[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price re-enters Donchian Bands (mean reversion) OR trend turns down
            if curr_close < dc_middle[i] or not uptrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price re-enters Donchian Bands (mean reversion) OR trend turns up
            if curr_close > dc_middle[i] or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals