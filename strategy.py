#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1w trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) and price above 1w EMA(50) with volume > 1.5x 20-day average.
# Short when Williams %R > -20 (overbought) and price below 1w EMA(50) with volume > 1.5x 20-day average.
# Exit when Williams %R crosses back above -50 (for long) or below -50 (for short).
# Williams %R identifies mean-reversion opportunities in ranging markets.
# 1w EMA(50) filters trades to align with higher timeframe trend.
# Volume surge confirms conviction behind the move.
# Designed for ~10-20 trades/year per symbol to minimize fee drag.
name = "1d_WilliamsR_1wEMA50_Volume_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # EMA(50) on 1w close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: current volume > 1.5 * 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        wr_val = williams_r[i]
        ema_val = ema_50_1w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: oversold + above 1w EMA + volume surge
            if wr_val < -80 and close_val > ema_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: overbought + below 1w EMA + volume surge
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