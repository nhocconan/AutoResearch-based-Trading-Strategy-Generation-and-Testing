#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with weekly trend filter
# - Uses 1w EMA(34) for trend direction (long when price > EMA, short when price < EMA)
# - Uses 6h Williams %R(14) for mean reversion entries (long when %R < -80, short when %R > -20)
# - Requires volume > 1.3 * 50-period volume average for confirmation
# - Fixed position size 0.25 to manage drawdown and reduce fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in bull markets via mean reversion in uptrend, in bear via mean reversion in downtrend

name = "6h_1w_williamsr_meanrev_trend_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Pre-compute 6h Williams %R(14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50.0  # neutral when range is zero
    )
    
    # Pre-compute volume confirmation: volume > 1.3 * 50-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1w EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions: mean reversion or trend change
            if williams_r[i] > -20:  # Overbought exit
                position = 0
                signals[i] = 0.0
            elif not uptrend:  # Trend changed to downtrend or sideways
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: mean reversion or trend change
            if williams_r[i] < -80:  # Oversold exit
                position = 0
                signals[i] = 0.0
            elif not downtrend:  # Trend changed to uptrend or sideways
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries in direction of 1w trend with volume confirmation
            if uptrend and williams_r[i] < -80 and volume_confirm[i]:  # Oversold in uptrend
                position = 1
                signals[i] = 0.25
            elif downtrend and williams_r[i] > -20 and volume_confirm[i]:  # Overbought in downtrend
                position = -1
                signals[i] = -0.25
    
    return signals