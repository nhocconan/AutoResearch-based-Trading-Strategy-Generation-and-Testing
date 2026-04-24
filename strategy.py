#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and volume confirmation.
- Williams Alligator: Jaw (EMA13, offset 8), Teeth (EMA8, offset 5), Lips (EMA5, offset 3)
- Long when Lips > Teeth > Jaw AND price > 1w EMA50 AND volume > 1.5 * volume MA20
- Short when Lips < Teeth < Jaw AND price < 1w EMA50 AND volume > 1.5 * volume MA20
- Exit when Alligator reverses (Teeth crosses Lips) OR volume drops below average
- Designed to catch strong trends in 12h timeframe with trend and volume filters
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w OHLC for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator components (using 12h data)
    # Jaw: EMA13, offset 8 bars
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.concatenate([np.full(8, np.nan), jaw[:-8]]) if len(jaw) > 8 else np.full_like(jaw, np.nan)
    
    # Teeth: EMA8, offset 5 bars
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.concatenate([np.full(5, np.nan), teeth[:-5]]) if len(teeth) > 5 else np.full_like(teeth, np.nan)
    
    # Lips: EMA5, offset 3 bars
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.concatenate([np.full(3, np.nan), lips[:-3]]) if len(lips) > 3 else np.full_like(lips, np.nan)
    
    # Volume confirmation: volume > 1.5 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13, 8, 5, 20)  # Need 1w EMA50, Alligator components, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (Alligator bullish alignment) AND uptrend AND high volume
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (Alligator bearish alignment) AND downtrend AND high volume
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator reverses (Teeth crosses below Lips) OR volume drops below average
            if teeth[i] < lips[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator reverses (Teeth crosses above Lips) OR volume drops below average
            if teeth[i] > lips[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0