#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 12h EMA trend filter with Bollinger Band mean reversion.
- Calculate Bollinger Bands (20, 2.0) on 4h close
- Enter long when price touches lower BB AND closes above it AND price > 12h EMA34 (uptrend filter)
- Enter short when price touches upper BB AND closes below it AND price < 12h EMA34 (downtrend filter)
- Exit when price crosses the 20-period SMA (middle band)
- Fixed position size 0.25 to manage drawdown
- Uses Bollinger Band squeeze (bandwidth < 50th percentile) to avoid choppy markets
- Designed for 4h timeframe with strict entry conditions to limit trades to 75-200 total over 4 years
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12-hour data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Bollinger Bands (20, 2.0) on 4h data
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_band = sma_20 + 2.0 * std_20
    lower_band = sma_20 - 2.0 * std_20
    middle_band = sma_20  # 20-period SMA
    
    # Bollinger Band width for squeeze filter (avoid chop)
    bb_width = (upper_band - lower_band) / middle_band
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).rank(pct=True)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for BB width percentile
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(upper_band.iloc[i]) or np.isnan(lower_band.iloc[i]) or 
            np.isnan(middle_band.iloc[i]) or np.isnan(bb_width_percentile.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_band.iloc[i]
        lower = lower_band.iloc[i]
        middle = middle_band.iloc[i]
        bb_width_pct = bb_width_percentile.iloc[i]
        ema_val = ema_34_aligned[i]
        
        # Only trade when Bollinger Bands are not squeezed (avoid chop)
        # Trade when bandwidth is above 30th percentile (not too tight)
        if bb_width_pct < 0.30:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for mean reversion at Bollinger Bands with trend filter
            # Long: price touches lower BB AND closes above it AND price > 12h EMA34
            if low[i] <= lower and close[i] > lower and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price touches upper BB AND closes below it AND price < 12h EMA34
            elif high[i] >= upper and close[i] < upper and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses above middle band (mean reversion complete)
            if close[i] > middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses below middle band (mean reversion complete)
            if close[i] < middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_BollingerMeanReversion_12hEMA34"
timeframe = "4h"
leverage = 1.0