#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Trix_Momentum_VolumeSpike_12hTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 15:
        return np.zeros(n)
    
    # Calculate TRIX (15-period EMA of EMA of EMA of close, then 1-period percent change)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    trix = ema3.pct_change() * 100  # Convert to percentage
    
    # Get 12h EMA20 for trend filter
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_12h = (close_12h > ema20_12h).astype(float)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for TRIX and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trix.iloc[i]) if hasattr(trix, 'iloc') else np.isnan(trix[i]) or 
            np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trix_val = trix.iloc[i] if hasattr(trix, 'iloc') else trix[i]
        
        if position == 0:
            # Long entry: TRIX crosses above zero with volume spike and 12h uptrend
            long_cond = (trix_val > 0 and 
                         (i == start_idx or (trix.iloc[i-1] if hasattr(trix, 'iloc') else trix[i-1]) <= 0) and
                         vol_spike[i] and trend_12h_aligned[i] > 0.5)
            
            # Short entry: TRIX crosses below zero with volume spike and 12h downtrend
            short_cond = (trix_val < 0 and 
                          (i == start_idx or (trix.iloc[i-1] if hasattr(trix, 'iloc') else trix[i-1]) >= 0) and
                          vol_spike[i] and trend_12h_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero (momentum reversal)
            if trix_val < 0 and (i == start_idx or (trix.iloc[i-1] if hasattr(trix, 'iloc') else trix[i-1]) >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero (momentum reversal)
            if trix_val > 0 and (i == start_idx or (trix.iloc[i-1] if hasattr(trix, 'iloc') else trix[i-1]) <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX momentum crossover strategy with volume spike confirmation and 12h EMA20 trend filter on 4h timeframe.
# Enters long when TRIX crosses above zero (bullish momentum) with volume spike and 12h uptrend (close > EMA20).
# Enters short when TRIX crosses below zero (bearish momentum) with volume spike and 12h downtrend (close < EMA20).
# Exits when TRIX crosses back through zero, signaling momentum reversal.
# Uses 20-period volume MA with 2.0x threshold for volume confirmation to avoid false signals.
# Targets 20-30 trades/year on 4h timeframe to minimize fee drag while capturing momentum shifts.
# Works in both bull and bear markets by following the 12h trend while using TRIX for timely entries.
# Discrete sizing (0.25) minimizes churn from small signal fluctuations.