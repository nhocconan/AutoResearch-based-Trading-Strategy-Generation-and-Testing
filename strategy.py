#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) and price > 1d EMA(50) and volume > 1.5x 20-period average.
# Short when Williams %R > -20 (overbought) and price < 1d EMA(50) and volume > 1.5x 20-period average.
# Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts).
# Williams %R identifies reversal points in ranging markets, while 1d EMA filters for trend direction.
# Volume surge confirms conviction. Designed for ~20-40 trades/year per symbol.
name = "6h_WilliamsR_1dEMA50_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        wr_val = williams_r[i]
        ema_val = ema_50_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: oversold, above 1d EMA, with volume surge
            if wr_val < -80 and close_val > ema_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: overbought, below 1d EMA, with volume surge
            elif wr_val > -20 and close_val < ema_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses back above -50
            if wr_val > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses back below -50
            if wr_val < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals