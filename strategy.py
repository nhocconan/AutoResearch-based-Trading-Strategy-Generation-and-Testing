#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA34 trend filter + volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trending vs ranging markets
# 1d EMA34 ensures alignment with daily trend direction for institutional bias
# Volume spike (2.0x 20-period average) confirms participation
# Discrete position sizing (0.25) minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) for 12h timeframe
# Works in bull markets via Alligator alignment with daily trend and in bear markets via filtered signals

name = "12h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Williams Alligator on 12h data (periods: 13, 8, 5 with offsets)
    # Jaw (Blue): 13-period SMMA, offset 8 bars
    # Teeth (Red): 8-period SMMA, offset 5 bars
    # Lips (Green): 5-period SMMA, offset 3 bars
    def smma(src, period):
        """Smoothed Moving Average"""
        result = np.full_like(src, np.nan, dtype=float)
        if len(src) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(src[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(src)):
            result[i] = (result[i-1] * (period-1) + src[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply offsets (shift right by offset bars)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set NaN for rolled values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator and volume MA)
    start_idx = max(20, 13+8)  # buffer for 20-period MA and Alligator
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips[i] > teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] < jaw[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator uptrend + 1d close > EMA34 + volume spike
            if (alligator_long and 
                close[i] > ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend + 1d close < EMA34 + volume spike
            elif (alligator_short and 
                  close[i] < ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator trend breaks or 1d trend breaks
            if not alligator_long or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator trend breaks or 1d trend breaks
            if not alligator_short or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals