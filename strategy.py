#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator system with 1w trend filter and volume confirmation.
- Primary timeframe: 1d for lower trade frequency and reduced fee drag.
- HTF: 1w EMA13 for trend direction (bullish if close > EMA13, bearish if close < EMA13).
- Williams Alligator: Jaw (EMA13, 8-period shift), Teeth (EMA8, 5-period shift), Lips (EMA5, 3-period shift).
- Volume: Current 1d volume > 1.5 * 20-period volume MA to confirm institutional participation.
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND 1w EMA13 bullish AND volume spike.
         Short when Lips < Teeth < Jaw (bearish alignment) AND 1w EMA13 bearish AND volume spike.
- Exit: Opposite Alligator alignment (Teeth crosses Lips) or loss of volume confirmation.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
This combines the Williams Alligator's trend-following capability with higher timeframe filtering
and volume confirmation to avoid false signals. Works in both bull and bear markets by
only taking trades in the direction of the 1w trend, reducing whipsaw during sideways periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator components
    # Jaw: EMA13, 8-period shift
    jaw = pd.Series(close).ewm(span=13, adjust=False).mean().shift(8).values
    # Teeth: EMA8, 5-period shift
    teeth = pd.Series(close).ewm(span=8, adjust=False).mean().shift(5).values
    # Lips: EMA5, 3-period shift
    lips = pd.Series(close).ewm(span=5, adjust=False).mean().shift(3).values
    
    # Get 1w data for EMA13 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Calculate 1w EMA13
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 20-period 1w volume MA
    df_1w_volume = df_1w['volume'].values
    vol_ma_1w = pd.Series(df_1w_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1d
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Volume confirmation: current 1d volume > 1.5 * 20-period 1w volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_1w_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 8, 5, 20)  # Need enough bars for Alligator components and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Lips > Teeth > Jaw AND 1w EMA13 bullish (close > EMA)
                if bullish_alignment and close[i] > ema_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Lips < Teeth < Jaw AND 1w EMA13 bearish (close < EMA)
                elif bearish_alignment and close[i] < ema_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Bearish alignment (Teeth crosses below Lips) OR loss of volume confirmation
            if bearish_alignment or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish alignment (Lips crosses above Teeth) OR loss of volume confirmation
            if bullish_alignment or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_1wEMA13_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0