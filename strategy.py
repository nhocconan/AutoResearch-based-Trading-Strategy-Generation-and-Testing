#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with daily volume confirmation and trend filter
# Bollinger Band squeeze indicates low volatility and impending breakout.
# Breakout direction determined by daily EMA trend filter to avoid false breakouts.
# Volume confirmation ensures institutional participation.
# Works in bull/bear by using daily EMA trend (long above EMA, short below EMA).
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Bollinger Bands (20, 2) on 6h
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Bollinger Band width for squeeze detection
    bb_width = (bb_upper - bb_lower) / bb_middle
    # Squeeze: BB width below 20-period average width
    bb_width_series = pd.Series(bb_width)
    bb_width_avg = bb_width_series.rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_avg
    
    # Daily volume confirmation: volume > 1.5x average daily volume (20-period)
    vol_1d_series = pd.Series(volume_1d)
    avg_vol_1d = vol_1d_series.rolling(window=20, min_periods=20).mean().shift(1).values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40  # for BB and averages
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Look for breakout after squeeze
            if squeeze[i]:
                # Long: break above upper band with volume filter AND above daily EMA50
                if (price > bb_upper[i] and 
                    vol > 1.5 * avg_vol_1d_aligned[i] and 
                    price > ema_50_1d_aligned[i]):
                    position = 1
                    signals[i] = position_size
                # Short: break below lower band with volume filter AND below daily EMA50
                elif (price < bb_lower[i] and 
                      vol > 1.5 * avg_vol_1d_aligned[i] and 
                      price < ema_50_1d_aligned[i]):
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below middle band OR below daily EMA50
            if price < bb_middle[i] or price < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above middle band OR above daily EMA50
            if price > bb_middle[i] or price > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Bollinger_Squeeze_Breakout_EMA_Volume"
timeframe = "6h"
leverage = 1.0