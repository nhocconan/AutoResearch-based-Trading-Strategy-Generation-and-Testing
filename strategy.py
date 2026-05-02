#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator breakout with 1d Elder Ray filter and volume spike
# Williams Alligator (JAW=13, TEETH=8, LIPS=5 SMMA) identifies trend direction and absence of trend
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength
# Volume spike confirms breakout validity
# Works in bull markets (Alligator mouth up, Bull Power > 0) and bear markets (Alligator mouth down, Bear Power > 0)
# Uses discrete position sizing (0.25) to balance return and drawdown control
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag

name = "12h_WilliamsAlligator_1dElderRay_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Elder Ray filter (EMA13 and power calculations)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power_1d = df_1d['high'].values - ema_13_1d
    bear_power_1d = ema_13_1d - df_1d['low'].values
    
    # Align Elder Ray to 12h timeframe (wait for 1d bar to close)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Williams Alligator on 12h timeframe
    # SMMA (Smoothed Moving Average) calculation: first value = SMA, subsequent = (prev*(period-1) + current) / period
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=np.float64)
        result = np.full_like(data, np.nan, dtype=np.float64)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Volume confirmation (volume spike > 2.0 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Alligator lines)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine Alligator trend: Mouth open (all lines separated and ordered)
        # Uptrend: Lips > Teeth > Jaw (green > red > blue)
        # Downtrend: Jaw > Teeth > Lips (blue > red > green)
        # No trend: lines intertwined
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        
        uptrend = lips_val > teeth_val and teeth_val > jaw_val
        downtrend = jaw_val > teeth_val and teeth_val > lips_val
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator mouth up (uptrend) + Bull Power > 0 + volume confirmation
            if uptrend and bull_power_aligned[i] > 0 and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator mouth down (downtrend) + Bear Power > 0 + volume confirmation
            elif downtrend and bear_power_aligned[i] > 0 and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator mouth closes (no trend) OR Bear Power becomes positive (reversal signal)
            if not uptrend or bear_power_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator mouth closes (no trend) OR Bull Power becomes positive (reversal signal)
            if not downtrend or bull_power_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals