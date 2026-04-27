#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA trend filter and volume confirmation.
# Williams Alligator (JAWS/TEETH/LIPS) identifies trending vs ranging markets.
# When JAWS > TEETH > LIPS = strong uptrend; JAWS < TEETH < LIPS = strong downtrend.
# When lines intertwine = ranging market (no trade).
# Strategy: Trade only in strong trends (JAWS/TEETH/LIPS aligned) with pullback entries.
# Use 1d EMA34 as higher timeframe trend filter for confirmation.
# Volume spike confirms institutional participation.
# Designed for ~20-30 trades/year per symbol to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 12h chart
    # JAWS (13-period SMMA, 8-period offset)
    # TEETH (8-period SMMA, 5-period offset)
    # LIPS (5-period SMMA, 3-period offset)
    def smoothed_mma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan)
        sma = pd.Series(data).rolling(window=period, min_periods=period).mean().values
        mma = np.full_like(data, np.nan, dtype=float)
        mma[period-1] = sma[period-1]
        for i in range(period, len(data)):
            mma[i] = (mma[i-1] * (period-1) + data[i]) / period
        return mma
    
    jaws = smoothed_mma(close, 13)
    teeth = smoothed_mma(close, 8)
    lips = smoothed_mma(close, 5)
    
    # Apply offsets (shift right)
    jaws = np.roll(jaws, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Strong uptrend: JAWS > TEETH > LIPS
        if jaws[i] > teeth[i] > lips[i]:
            # Only trade in uptrend direction
            if close[i] > ema34_1d_aligned[i] and volume_filter[i]:
                # Buy on pullback to TEETH (8-period SMMA)
                if close[i] <= teeth[i] * 1.005 and close[i] >= teeth[i] * 0.995:
                    signals[i] = 0.25
                    position = 1
                elif position == 1:
                    # Hold long position
                    signals[i] = 0.25
                else:
                    signals[i] = 0.0
            else:
                # No trend alignment or no volume
                if position == 1:
                    signals[i] = 0.0  # exit long
                    position = 0
                else:
                    signals[i] = 0.0
        
        # Strong downtrend: JAWS < TEETH < LIPS
        elif jaws[i] < teeth[i] < lips[i]:
            # Only trade in downtrend direction
            if close[i] < ema34_1d_aligned[i] and volume_filter[i]:
                # Sell on rally to TEETH (8-period SMMA)
                if close[i] >= teeth[i] * 0.995 and close[i] <= teeth[i] * 1.005:
                    signals[i] = -0.25
                    position = -1
                elif position == -1:
                    # Hold short position
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            else:
                # No trend alignment or no volume
                if position == -1:
                    signals[i] = 0.0  # exit short
                    position = 0
                else:
                    signals[i] = 0.0
        
        # Ranging market: Alligator sleeping (lines intertwined)
        else:
            # No trade in ranging market
            if position != 0:
                signals[i] = 0.0  # exit any position
                position = 0
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0