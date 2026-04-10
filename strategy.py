#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversion with Weekly Trend Filter
# - Primary: 6h timeframe for balanced trade frequency (~12-37/year target)
# - HTF: 1w for trend direction (avoid counter-trend trades in strong trends)
# - Long: Williams %R(14) < -80 (oversold) + price > weekly EMA20 (uptrend)
# - Short: Williams %R(14) > -20 (overbought) + price < weekly EMA20 (downtrend)
# - Exit: Williams %R returns to -50 level (mean reversion) or opposite extreme
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: Weekly EMA20 filter ensures trades align with major trend,
#   Williams %R captures mean reversion within trend. Avoids whipsaws in ranging markets.

name = "6h_1w_williamsr_extreme_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 6h Williams %R(14)
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 1w EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R oversold (< -80) + price above weekly EMA20 (uptrend)
            if (williams_r[i] < -80 and close_6h[i] > ema_20_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R overbought (> -20) + price below weekly EMA20 (downtrend)
            elif (williams_r[i] > -20 and close_6h[i] < ema_20_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Williams %R returns to -50 level (mean reversion)
            # 2. Williams %R reaches opposite extreme (contrarian signal)
            
            if position == 1:  # Long position
                exit_condition = (
                    williams_r[i] > -50 or  # Returned to mean reversion level
                    williams_r[i] > -20     # Reached opposite extreme (overbought)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    williams_r[i] < -50 or  # Returned to mean reversion level
                    williams_r[i] < -80     # Reached opposite extreme (oversold)
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals