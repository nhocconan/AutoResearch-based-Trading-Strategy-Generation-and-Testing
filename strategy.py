#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1w EMA trend filter.
# Long when price breaks above 12h Donchian high (20-period) with volume > 2x 20-period average and price > 1w EMA(50).
# Short when price breaks below 12h Donchian low (20-period) with volume > 2x 20-period average and price < 1w EMA(50).
# Exit when price crosses the 12-period EMA of the 12h timeframe.
# Uses Donchian channels for structure, volume surge for conviction, weekly EMA for trend filter.
# Designed for ~15-30 trades/year per symbol.
name = "12h_Donchian_20_Volume_EMA50_TrendFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # EMA(50) on 1w close for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # 12h EMA(12) for exit
    ema_12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_12[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_filter = volume_filter[i]
        ema_exit = ema_12[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume surge and above weekly EMA
            if close_val > upper and vol_filter and close_val > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume surge and below weekly EMA
            elif close_val < lower and vol_filter and close_val < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 12-period EMA
            if close_val < ema_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 12-period EMA
            if close_val > ema_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals