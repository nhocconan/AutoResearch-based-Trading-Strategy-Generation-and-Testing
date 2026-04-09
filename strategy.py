#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion + 1w trend filter + volume spike
# Williams %R identifies oversold/overbought conditions on 6h
# 1w EMA(34) confirms higher timeframe trend direction for bias
# Volume spike (>2x 24-period average) confirms participation
# Discrete sizing 0.25 limits drawdown; targets 50-150 total trades over 4 years
# Works in bull/bear: mean reversion in ranges, trend filter avoids counter-trend in strong moves

name = "6h_1w_williamsr_ema_volume_v1"
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
    
    # Load 1w data ONCE before loop for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate 1w EMA(34)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 6h Williams %R(14)
    williams_r = np.full(n, np.nan)
    for i in range(n):
        if i < 14:
            williams_r[i] = np.nan
        else:
            highest_high = np.max(high[i-14:i+1])
            lowest_low = np.min(low[i-14:i+1])
            if highest_high == lowest_low:
                williams_r[i] = -50.0  # avoid division by zero
            else:
                williams_r[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    # Calculate 6h average volume (24-period = 6 days)
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 24:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_1w_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2x 24-period average
        volume_confirmed = volume[i] > 2.0 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) OR price < 1w EMA (trend change)
            if williams_r[i] > -20.0 or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R > -80 (oversold) OR price > 1w EMA (trend change)
            if williams_r[i] > -80.0 or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Williams %R extremes + 1w EMA filter
            if volume_confirmed:
                # Long entry: Williams %R < -80 (oversold) AND price > 1w EMA (bullish alignment)
                if williams_r[i] < -80.0 and close[i] > ema_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Williams %R > -20 (overbought) AND price < 1w EMA (bearish alignment)
                elif williams_r[i] > -20.0 and close[i] < ema_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals