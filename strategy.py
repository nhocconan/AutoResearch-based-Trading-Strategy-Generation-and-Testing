#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator + Elder Ray + 1d Fractal Confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for Williams fractal confirmation and 1w for EMA34 trend filter.
- Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) - smoothed with SMA.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13.
- Fractal Confirmation: 1d Williams fractals require 2-bar confirmation delay.
- Entry: Long when Lips > Teeth > Jaw AND Bull Power > 0 AND bullish fractal confirmed.
         Short when Lips < Teeth < Jaw AND Bear Power < 0 AND bearish fractal confirmed.
- Exit: Opposite Alligator alignment (Lips crosses Jaw) or power signal reversal.
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by aligning with Alligator trend and filtering with Elder Ray and fractals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 1d Williams fractals for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate Williams Alligator on 4h timeframe
    jaw_period = 13
    jaw_shift = 8
    teeth_period = 8
    teeth_shift = 5
    lips_period = 5
    lips_shift = 3
    
    # Jaw: SMA(13) shifted 8 bars ahead
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    jaw = np.roll(jaw, -jaw_shift)  # shift left (future)
    jaw[:jaw_shift] = np.nan  # fill shifted values with NaN
    
    # Teeth: SMA(8) shifted 5 bars ahead
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    teeth = np.roll(teeth, -teeth_shift)
    teeth[:teeth_shift] = np.nan
    
    # Lips: SMA(5) shifted 3 bars ahead
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().values
    lips = np.roll(lips, -lips_shift)
    lips[:lips_shift] = np.nan
    
    # Calculate Elder Ray on 4h timeframe
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(jaw_shift, teeth_period, lips_period, 13)  # Need sufficient data
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Alligator alignment: Lips > Teeth > Jaw for bullish, Lips < Teeth < Jaw for bearish
        bullish_alligator = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alligator = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray: Bull Power > 0 for bullish bias, Bear Power < 0 for bearish bias
        bullish_elder = bull_power[i] > 0
        bearish_elder = bear_power[i] < 0
        
        # Fractal confirmation: 1d Williams fractals require confirmation
        bullish_fractal = bullish_fractal_aligned[i] == 1
        bearish_fractal = bearish_fractal_aligned[i] == 1
        
        # Exit conditions: opposite Alligator alignment or power signal reversal
        if position != 0:
            # Exit long: bearish Alligator alignment or bearish Elder Ray
            if position == 1:
                if bearish_alligator or not bullish_elder:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bullish Alligator alignment or bullish Elder Ray
            elif position == -1:
                if bullish_alligator or not bearish_elder:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with Elder Ray and fractal confirmation
        if position == 0:
            # Long: bullish Alligator AND bullish Elder Ray AND bullish fractal
            long_condition = bullish_alligator and bullish_elder and bullish_fractal
            
            # Short: bearish Alligator AND bearish Elder Ray AND bearish fractal
            short_condition = bearish_alligator and bearish_elder and bearish_fractal
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_ElderRay_1dFractalConfirm_1wEMA34Trend_v1"
timeframe = "4h"
leverage = 1.0