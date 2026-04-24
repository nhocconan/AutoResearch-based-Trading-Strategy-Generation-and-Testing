#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume spike confirmation.
- Williams Alligator: Jaw (EMA13 of median price, 8-bar shift), Teeth (EMA8 of median price, 5-bar shift), Lips (EMA5 of median price, 3-bar shift)
- Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume > 2.0 * median volume of last 20 bars
- Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume > 2.0 * median volume of last 20 bars
- Exit when Alligator alignment breaks or price crosses 1d EMA50 in opposite direction
- Uses 4h primary timeframe with 1d HTF to target 75-200 total trades over 4 years (19-50/year)
- Williams Alligator identifies trending vs ranging markets: converged lines = ranging (no trade), diverged = trending (trade)
- 1d EMA50 ensures alignment with higher timeframe trend to avoid whipsaws
- Volume spike confirmation (>2.0x median) ensures momentum behind breakouts, reducing false signals
- Designed for BTC/ETH with edge in trending markets; avoids ranging markets where Alligator converges
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Williams Alligator components
    # Jaw: EMA13 of median price, 8-bar shift
    jaw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth: EMA8 of median price, 5-bar shift
    teeth = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips: EMA5 of median price, 3-bar shift
    lips = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (2.0 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 8, 5) + 8  # account for jaw shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long: bullish alignment, price above 1d EMA50, volume confirmation
            if bullish_alignment and close[i] > ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment, price below 1d EMA50, volume confirmation
            elif bearish_alignment and close[i] < ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish alignment OR price below 1d EMA50
            if bearish_alignment or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment OR price above 1d EMA50
            if bullish_alignment or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0