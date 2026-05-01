#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band breakout with 12h trend filter and volume confirmation.
# Long when price breaks above upper BB(20,2) AND 12h EMA50 uptrend AND volume > 1.5x 20-period median.
# Short when price breaks below lower BB(20,2) AND 12h EMA50 downtrend AND volume > 1.5x 20-period median.
# Bollinger Bands capture volatility expansion; 12h EMA50 filters for higher-timeframe trend alignment; volume spike confirms breakout conviction.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.

name = "6h_Bollinger_Breakout_12hEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands (20,2)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate 12h EMA50 trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for Bollinger Bands, EMA, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 12h EMA50 direction
        uptrend = curr_close > ema_50_12h_aligned[i]
        downtrend = curr_close < ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper BB AND uptrend AND volume spike
            if curr_close > bb_upper[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below lower BB AND downtrend AND volume spike
            elif curr_close < bb_lower[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price re-enters Bollinger Bands (mean reversion) OR trend turns down
            if curr_close < bb_middle[i] or not uptrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price re-enters Bollinger Bands (mean reversion) OR trend turns up
            if curr_close > bb_middle[i] or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals