#!/usr/bin/env python3
"""
1d_Williams_Alligator_JawTeethLips_1wTrend_VolumeSpike
Hypothesis: Daily Williams Alligator (Jaw/Teeth/Lips) with 1-week trend filter (price > 1w EMA34) and volume confirmation (>2.0x 20-period average).
Long when Lips cross above Teeth and Jaw in 1-week uptrend with volume confirmation.
Short when Lips cross below Teeth and Jaw in 1-week downtrend with volume confirmation.
Exit via opposite Alligator lines or ATR trailing stop (2.5*ATR from extreme).
Alligator acts as trend-following system that stays out of choppy markets via convergence/divergence.
Volume confirmation ensures breakouts have conviction. 1-week trend filter aligns with higher timeframe bias.
Designed for ~30-80 trades over 4 years (7-20/year) via tight Alligator crossover conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:  # need 34 for EMA
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Alligator calculation (same timeframe as prices)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:  # need 13 for Lips
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Alligator on 1d data
    # Jaw (blue line): 13-period SMMA shifted 8 bars
    jaw_period = 13
    jaw_shift = 8
    sma_jaw = pd.Series(close_1d).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    jaw = np.roll(sma_jaw, jaw_shift)
    jaw[:jaw_shift] = np.nan  # first 8 values invalid due to shift
    
    # Teeth (red line): 8-period SMMA shifted 5 bars
    teeth_period = 8
    teeth_shift = 5
    sma_teeth = pd.Series(close_1d).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    teeth = np.roll(sma_teeth, teeth_shift)
    teeth[:teeth_shift] = np.nan  # first 5 values invalid due to shift
    
    # Lips (green line): 5-period SMMA shifted 3 bars
    lips_period = 5
    lips_shift = 3
    sma_lips = pd.Series(close_1d).rolling(window=lips_period, min_periods=lips_period).mean().values
    lips = np.roll(sma_lips, lips_shift)
    lips[:lips_shift] = np.nan  # first 3 values invalid due to shift
    
    # ATR for stoploss (21-period)
    atr_period = 21
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0  # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(100, atr_period, 20, jaw_shift, teeth_shift, lips_shift)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_34_1w_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (1w EMA34 filter)
            if close[i] > ema_trend:  # 1w uptrend regime
                # Long: Lips cross above Teeth and Jaw with volume confirmation
                lips_above_teeth = lips[i] > teeth[i]
                lips_above_jaw = lips[i] > jaw[i]
                long_signal = lips_above_teeth and lips_above_jaw and vol_regime[i]
            else:  # 1w downtrend regime
                # Short: Lips cross below Teeth and Jaw with volume confirmation
                lips_below_teeth = lips[i] < teeth[i]
                lips_below_jaw = lips[i] < jaw[i]
                short_signal = lips_below_teeth and lips_below_jaw and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update extreme for trailing stop
            if close[i] > long_extreme:
                long_extreme = close[i]
            # Exit conditions: 
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = long_extreme - 2.5 * atr[i]
            # 2. Lips cross below Jaw (trend weakening)
            if close[i] <= atr_stop or lips[i] < jaw[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update extreme for trailing stop
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit conditions:
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = short_extreme + 2.5 * atr[i]
            # 2. Lips cross above Jaw (trend weakening)
            if close[i] >= atr_stop or lips[i] > jaw[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Williams_Alligator_JawTeethLips_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0