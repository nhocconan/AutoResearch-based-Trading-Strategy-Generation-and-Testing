#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume spike
# - Uses 1d EMA(50) to determine trend: price > EMA50 = uptrend, price < EMA50 = downtrend
# - In uptrend: look for Williams %R(14) < -80 (oversold) for long entries
# - In downtrend: look for Williams %R(14) > -20 (overbought) for short entries
# - Volume confirmation: current 6h volume > 2.0x 24-period average to avoid false signals
# - Fixed position size 0.25 to control drawdown
# - Williams %R identifies exhaustion points, EMA filter ensures trading with higher timeframe trend
# - Target: 12-25 trades/year on 6h timeframe (48-100 total over 4 years)

name = "6h_1d_williamsr_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 6h data
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Pre-compute volume confirmation (24-period average)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_ma_24[i]) or vol_ma_24[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2.0x average
        volume_confirmed = volume[i] > 2.0 * vol_ma_24[i]
        
        if position == 1:  # Long position
            # Exit long when Williams %R rises above -50 (exiting oversold)
            if williams_r[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short when Williams %R falls below -50 (exiting overbought)
            if williams_r[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Williams %R extreme + 1d trend filter + volume confirmation
            if volume_confirmed:
                # Determine trend from 1d EMA50
                uptrend = close[i] > ema_50_aligned[i]
                downtrend = close[i] < ema_50_aligned[i]
                
                # Long entry: oversold in uptrend
                if uptrend and williams_r[i] < -80:
                    position = 1
                    signals[i] = 0.25
                # Short entry: overbought in downtrend
                elif downtrend and williams_r[i] > -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals