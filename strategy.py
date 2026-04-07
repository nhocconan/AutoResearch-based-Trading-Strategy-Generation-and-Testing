#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray (Bull/Bear Power) + 12h Trend Filter + Volume Spike
# Hypothesis: Elder Ray measures bull/bear power via EMA13. In strong trends (12h EMA50),
# we take trades in direction of 12h trend when Bull/Bear power confirms momentum
# and volume spikes (>1.5x 20-period average) indicate institutional participation.
# Works in bull/bear markets by following 12h trend with momentum/volume confirmation.
# Target: 12-37 trades/year (50-150 total over 4 years) for 6h timeframe.

name = "6h_elder_ray_12h_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # EMA(13) for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull power: high - EMA13
    bear_power = low - ema_13   # Bear power: low - EMA13
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: 12h trend turns bearish or bull power fails
            if close[i] < ema_50_12h_aligned[i] or bull_power[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: 12h trend turns bullish or bear power fails
            if close[i] > ema_50_12h_aligned[i] or bear_power[i] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Strong uptrend on 12h + bull power positive
                if close[i] > ema_50_12h_aligned[i] and bull_power[i] > 0:
                    position = 1
                    signals[i] = 0.25
                # Strong downtrend on 12h + bear power negative
                elif close[i] < ema_50_12h_aligned[i] and bear_power[i] < 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals