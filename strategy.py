#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume confirmation
# - Uses 12h HTF for trend direction (price > EMA50 = uptrend, < EMA50 = downtrend)
# - 6h Williams %R(14) for mean reversion signals: long when %R < -80 (oversold), short when %R > -20 (overbought)
# - Only take mean reversion trades in direction of 12h trend (avoid counter-trend whipsaws)
# - Volume confirmation: current 6h volume > 1.5x 20-period average to avoid low-volume false signals
# - Fixed position size 0.25 to control drawdown
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years)
# - Williams %R works well in ranging markets (common in 2025 BTC/ETH bear/range) and catches reversals in trends

name = "6h_12h_williamsr_mean_reversion_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h timeframe (wait for completed 12h bar)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close) / (highest_high_14 - lowest_low_14) * -100
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0 or
            np.isnan(highest_high_14[i]) or np.isnan(lowest_low_14[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: 12h price > EMA50 = uptrend, < EMA50 = downtrend
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Williams %R levels
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when Williams %R returns above -50 (mean reversion complete) or trend changes
            if williams_r[i] > -50 or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when Williams %R returns below -50 (mean reversion complete) or trend changes
            if williams_r[i] < -50 or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Mean reversion entry with volume confirmation and trend alignment
            if volume_confirmed:
                # Long: oversold in uptrend
                if uptrend and oversold:
                    position = 1
                    signals[i] = position_size
                # Short: overbought in downtrend
                elif downtrend and overbought:
                    position = -1
                    signals[i] = -position_size
    
    return signals