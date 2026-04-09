#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d volume spike and 1w trend filter
# In strong 1w uptrend: buy Williams %R oversold (< -80) with volume confirmation
# In strong 1w downtrend: sell Williams %R overbought (> -20) with volume confirmation
# In ranging 1w (no clear trend): avoid trading to prevent whipsaws
# Uses discrete position sizing 0.25 to limit trades and reduce fee drag
# Williams %R identifies exhaustion points in trends, volume confirms participation

name = "4h_1w_williamsr_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) and EMA(200) for trend determination
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Determine 1w trend regime: 1 = uptrend, -1 = downtrend, 0 = ranging/no trend
    trend_1w = np.zeros(len(close_1w))
    trend_1w[close_1w > ema_50_1w] = 1
    trend_1w[close_1w < ema_50_1w] = -1
    # Only consider strong trend when price is also above/below ema_200
    strong_uptrend = (close_1w > ema_50_1w) & (close_1w > ema_200_1w)
    strong_downtrend = (close_1w < ema_50_1w) & (close_1w < ema_200_1w)
    trend_regime = np.zeros(len(close_1w))
    trend_regime[strong_uptrend] = 1   # Strong uptrend
    trend_regime[strong_downtrend] = -1 # Strong downtrend
    # 0 = ranging or weak trend (avoid trading)
    
    # Align 1w trend regime to 4h timeframe
    trend_regime_aligned = align_htf_to_ltf(prices, df_1w, trend_regime)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d average volume (20-period)
    volume_1d_series = pd.Series(volume_1d)
    avg_volume_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d average volume to 4h timeframe
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    volume_confirmed = volume > 1.5 * avg_volume_1d_aligned
    
    # Calculate Williams %R on 4h data (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high_14 - lowest_low_14) != 0,
        ((highest_high_14 - close) / (highest_high_14 - lowest_low_14)) * -100,
        -50  # Neutral when range is zero
    )
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(trend_regime_aligned[i]) or np.isnan(volume_confirmed[i]) or
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when there's a strong 1w trend (avoid ranging markets)
        if trend_regime_aligned[i] == 0:
            # Flat ranging market - exit any position and stay flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if Williams %R rises above -50 (exhaustion) or trend changes
            if williams_r[i] > -50 or trend_regime_aligned[i] != 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if Williams %R falls below -50 (exhaustion) or trend changes
            if williams_r[i] < -50 or trend_regime_aligned[i] != -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if trend_regime_aligned[i] == 1:  # Strong 1w uptrend
                # Enter long when Williams %R is oversold (< -80) with volume confirmation
                if williams_r[i] < -80 and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
            elif trend_regime_aligned[i] == -1:  # Strong 1w downtrend
                # Enter short when Williams %R is overbought (> -20) with volume confirmation
                if williams_r[i] > -20 and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals