#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R Reversal with 1w Trend Filter and Volume Confirmation
# Uses Williams %R (14-period) from 1d for mean-reversion signals
# 1w EMA (50) provides trend direction filter to avoid counter-trend trades
# Volume confirmation (>1.5x average) ensures institutional participation
# Williams %R identifies overbought/oversold conditions: >-20 = overbought, <-80 = oversold
# In trending markets (price > 1w EMA), we look for oversold bounces (long)
# In ranging markets, we trade reversals from extremes
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA (50) for trend direction
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Load 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for EMA and Williams %R
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: price above/below 1w EMA
        above_ema = price > ema_1w_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with volume filter
            if williams_r_aligned[i] < -80 and vol > 1.5 * avg_vol[i]:
                # In uptrend, take oversold bounce
                # In downtrend or ranging, still take mean reversion
                position = 1
                signals[i] = position_size
            # Short: Williams %R overbought (> -20) with volume filter
            elif williams_r_aligned[i] > -20 and vol > 1.5 * avg_vol[i]:
                # In downtrend, take overbought rejection
                # In uptrend or ranging, still take mean reversion
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to overbought (> -20) or reverse signal
            if williams_r_aligned[i] > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to oversold (< -80) or reverse signal
            if williams_r_aligned[i] < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_WilliamsR_1wEMA_Volume"
timeframe = "1d"
leverage = 1.0