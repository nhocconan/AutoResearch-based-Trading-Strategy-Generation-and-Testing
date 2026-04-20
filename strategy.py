# Hypothetical: 1D-1W Donchian Breakout with Volume Confirmation and 1-Week Trend Filter
# Designed for 1d timeframe with weekly trend filter to avoid counter-trend trades.
# Uses Donchian(20) breakout on daily close, volume > 1.5x 20-day average for confirmation,
# and trades only in direction of weekly EMA50 trend to improve win rate in both bull/bear markets.
# Target: 20-80 total trades over 4 years (5-20/year) to minimize fee drag.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Donchian breakout and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily Donchian channels (20-period)
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        
        if position == 0:
            # Long: price breaks above 20-day high, above weekly EMA50 (uptrend), with volume confirmation
            if (price > high_max_20[i] and 
                price > ema50_1w_aligned[i] and 
                vol > 1.5 * vol_ma_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low, below weekly EMA50 (downtrend), with volume confirmation
            elif (price < low_min_20[i] and 
                  price < ema50_1w_aligned[i] and 
                  vol > 1.5 * vol_ma_20[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 20-day low or reverses below weekly EMA50
            if (price < low_min_20[i] or 
                price < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 20-day high or reverses above weekly EMA50
            if (price > high_max_20[i] or 
                price > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Donchian20_TrendFilter_VolumeConfirm"
timeframe = "1d"
leverage = 1.0