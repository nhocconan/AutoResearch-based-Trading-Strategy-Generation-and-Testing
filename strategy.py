#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation.
- Williams Alligator: Jaw (EMA13, 8-period shift), Teeth (EMA8, 5-period shift), Lips (EMA5, 3-period shift)
- Long when Lips > Teeth > Jaw (bullish alignment) AND 1w EMA34 > 1w EMA89 (long-term uptrend) AND volume > 2.0 * 20-period average
- Short when Lips < Teeth < Jaw (bearish alignment) AND 1w EMA34 < 1w EMA89 (long-term downtrend) AND volume > 2.0 * 20-period average
- Exit when Alligator alignment breaks (Lips crosses Teeth or Jaw) OR volume < 1.0 * 20-period average
- Uses 12h primary with 1w HTF for trend filter to avoid whipsaws in ranging markets
- Alligator identifies trend phases; 1w EMA filter ensures alignment with higher timeframe trend; volume confirms conviction
- Designed to work in both bull (bullish Alligator alignment) and bear (bearish Alligator alignment) markets
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year)
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
    
    # Calculate Williams Alligator components on 12h data
    # Jaw: EMA13 of median price, shifted 8 bars
    median_price = (high + low) / 2
    jaw_raw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)  # Shift 8 bars forward
    jaw[:8] = np.nan  # First 8 values invalid due to shift
    
    # Teeth: EMA8 of median price, shifted 5 bars
    teeth_raw = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)  # Shift 5 bars forward
    teeth[:5] = np.nan  # First 5 values invalid due to shift
    
    # Lips: EMA5 of median price, shifted 3 bars
    lips_raw = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)  # Shift 3 bars forward
    lips[:3] = np.nan  # First 3 values invalid due to shift
    
    # Calculate 1w EMA34 and EMA89 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 100:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1w = pd.Series(close_1w).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1w EMAs to 12h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    ema89_1w_aligned = align_htf_to_ltf(prices, df_1w, ema89_1w)
    
    # Trend filter: bullish if EMA34 > EMA89, bearish if EMA34 < EMA89
    bullish_trend = ema34_1w_aligned > ema89_1w_aligned
    bearish_trend = ema34_1w_aligned < ema89_1w_aligned
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    volume_exit = volume < (1.0 * vol_ma)  # Exit when volume drops below average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20) + 8  # Need Alligator components (with shifts) and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(ema89_1w_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND bullish trend AND volume confirmation
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and bullish_trend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) AND bearish trend AND volume confirmation
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and bearish_trend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw) OR volume exit
            if lips[i] <= teeth[i] or teeth[i] <= jaw[i] or volume_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw) OR volume exit
            if lips[i] >= teeth[i] or teeth[i] >= jaw[i] or volume_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wEMATrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0