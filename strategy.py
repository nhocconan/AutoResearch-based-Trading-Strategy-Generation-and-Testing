#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d EMA(50) trend filter and volume spike confirmation
# - Primary: 12h price breaks above Camarilla H3 (long) or below L3 (short) from prior 1d
# - Trend filter: 1d EMA(50) direction (price above/below EMA for bias)
# - Volume filter: 12h volume > 1.5x 20-period volume MA to confirm participation
# - Exit: Price retouches Camarilla pivot point (mean reversion in range)
# - Position sizing: 0.25 discrete level to minimize fee churn
# - Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# - Works in bull/bear: Camarilla levels work in ranging markets, EMA filter avoids counter-trend,
#   volume spike ensures institutional interest, pivot exit captures mean reversion

name = "12h_1d_camarilla_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior 1d (H3, L3, pivot)
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    # Pivot = (high + low + close)/3
    rng_1d = high_1d - low_1d
    H3_1d = close_1d + 1.1 * rng_1d / 4
    L3_1d = close_1d - 1.1 * rng_1d / 4
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    
    # Align Camarilla levels to 12h (use prior completed 1d)
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h volume spike filter: volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(H3_1d_aligned[i]) or np.isnan(L3_1d_aligned[i]) or
            np.isnan(pivot_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: break above H3 + price above 1d EMA(50) + volume spike
            if (close[i] > H3_1d_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: break below L3 + price below 1d EMA(50) + volume spike
            elif (close[i] < L3_1d_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot point (mean reversion)
            # Exit: price retouches pivot point
            if position == 1:  # Long position
                if close[i] <= pivot_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= pivot_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals