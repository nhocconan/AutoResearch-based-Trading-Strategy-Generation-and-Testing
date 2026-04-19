#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with daily trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND price > daily EMA200 AND volume > 1.5x daily average volume
# Short when price breaks below Donchian(20) low AND price < daily EMA200 AND volume > 1.5x daily average volume
# Exit when price crosses back below/above Donchian(20) mid-line (10-period average)
# Uses Donchian for breakout structure, daily EMA200 for trend filter, volume for confirmation.
# Target: 20-50 trades/year per symbol.
name = "4h_Donchian_Breakout_Trend_Volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA200 and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA200 for trend filter
    daily_ema200 = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    daily_ema200_aligned = align_htf_to_ltf(prices, df_1d, daily_ema200)
    
    # Daily average volume (20-period) for confirmation
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Donchian channels (20-period high/low)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # Ensure EMA200 and Donchian are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(daily_ema200_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        mid = donchian_mid[i]
        ema200 = daily_ema200_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        
        # Volume confirmation
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: price breaks above Donchian high AND price > EMA200 AND volume confirmation
            if price > upper and price > ema200 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND price < EMA200 AND volume confirmation
            elif price < lower and price < ema200 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian mid-line
            if price < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian mid-line
            if price > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals