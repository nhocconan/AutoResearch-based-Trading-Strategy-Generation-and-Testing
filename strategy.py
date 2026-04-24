#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Alligator with 1d trend filter and volume confirmation.
- Williams %R(14) from Alligator system: %R < -80 = oversold (long bias), %R > -20 = overbought (short bias)
- Alligator filter: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs of median price. Long when Lips > Teeth > Jaw (bullish alignment)
- 1d EMA50 trend filter: only take longs in bullish regime (close > EMA50), shorts in bearish regime (close < EMA50)
- Volume confirmation: current volume > 1.8 * 20-period average volume
- Exit on Williams %R crossing back above -50 (for longs) or below -50 (for shorts) to avoid whipsaw
- Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year)
- Williams %R provides mean reversion edge in ranging markets, Alligator filters trend, volume confirms momentum
- Signal size: 0.25 discrete levels to minimize fee churn
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
    
    # Williams %R calculation (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low + 1e-10)) * -100
    
    # Alligator components (using median price)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw).rolling(window=8, min_periods=8).mean().values  # Smoothed
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth).rolling(window=5, min_periods=5).mean().values  # Smoothed
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips).rolling(window=3, min_periods=3).mean().values  # Smoothed
    
    # Alligator alignment: bullish when Lips > Teeth > Jaw
    bullish_alligator = (lips > teeth) & (teeth > jaw)
    bearish_alligator = (lips < teeth) & (teeth < jaw)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    daily_close = df_1d['close'].values
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Trend filter: bullish if close > EMA50, bearish if close < EMA50
    bullish_regime = close > ema_50_1d_aligned
    bearish_regime = close < ema_50_1d_aligned
    
    # Volume confirmation: volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need Williams %R(14), Alligator, EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND bullish Alligator AND bullish regime AND volume confirmation
            if williams_r[i] < -80.0 and bullish_alligator[i] and bullish_regime[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND bearish Alligator AND bearish regime AND volume confirmation
            elif williams_r[i] > -20.0 and bearish_alligator[i] and bearish_regime[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50 (mean reversion)
            if williams_r[i] > -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (mean reversion)
            if williams_r[i] < -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Alligator_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0