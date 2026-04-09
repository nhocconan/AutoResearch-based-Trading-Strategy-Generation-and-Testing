#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume confirmation
# - Uses 1w HTF for trend direction (price > SMA50 = uptrend, < = downtrend)
# - 6h Williams %R(14) for mean reversion signals (long when < -80, short when > -20)
# - Volume confirmation: current 6h volume > 1.5x 20-period average
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Williams %R works well in ranging markets and captures reversals in bear markets

name = "6h_1w_williamsr_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w SMA for trend filter
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Align 1w SMA to 6h timeframe (wait for completed 1w bar)
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Calculate 6h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(sma_50_1w_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: 1w price > SMA50 = uptrend, < = downtrend
        uptrend = close[i] > sma_50_1w_aligned[i]
        downtrend = close[i] < sma_50_1w_aligned[i]
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when Williams %R > -20 (overbought) or trend change
            if williams_r[i] > -20 or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when Williams %R < -80 (oversold) or trend change
            if williams_r[i] < -80 or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Mean reversion entry with volume confirmation and trend alignment
            if volume_confirmed:
                # Long: Williams %R < -80 (oversold) in uptrend
                if uptrend and williams_r[i] < -80:
                    position = 1
                    signals[i] = position_size
                # Short: Williams %R > -20 (overbought) in downtrend
                elif downtrend and williams_r[i] > -20:
                    position = -1
                    signals[i] = -position_size
    
    return signals