# 6h_TurtleTrend_Signal
# Hypothesis: On 6h timeframe, use Turtle Trading breakout logic (20-period Donchian) combined with 1-day trend filter (price > 200 EMA for longs, < 200 EMA for shorts).
# This captures medium-term trends while avoiding counter-trend trades. Works in both bull and bear markets by only trading in direction of higher timeframe trend.
# Volume confirmation ensures breakouts have conviction. Designed for low trade frequency (~15-30/year) to minimize fee drag.
# Target: 50-120 total trades over 4 years per symbol.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_TurtleTrend_Signal"
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
    
    # Get daily data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily 200 EMA for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 6h Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 200)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        trend_ema = ema_200_1d_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: break above Donchian high with volume and above daily 200 EMA
            if price > upper and volume_confirmed and price > trend_ema:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume and below daily 200 EMA
            elif price < lower and volume_confirmed and price < trend_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below Donchian low (stop and reverse) or trend fails
            if price < lower or price < trend_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above Donchian high (stop and reverse) or trend fails
            if price > upper or price > trend_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals