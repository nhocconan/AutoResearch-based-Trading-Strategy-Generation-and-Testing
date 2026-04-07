#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Williams Alligator + Volume Spike + Chop Filter
# Hypothesis: Williams Alligator (Jaw/Teeth/Lips SMAs) identifies trend direction and strength.
# Combined with volume spikes to confirm momentum and Choppiness Index to filter ranging markets,
# this strategy captures strong trends while avoiding whipsaws. The Alligator's convergence/divergence
# provides clear entry/exit signals. Works in both bull and bear markets by following the dominant trend.
# Target: 15-25 trades/year to minimize fee drag on daily timeframe.
name = "1d_williams_alligator_volume_chop_v1"
timeframe = "1d"
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
    
    # Williams Alligator: three SMAs with different periods and shifts
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume spike confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Choppiness Index (14) - range detection
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros(n)
    for i in range(14, n):
        if max_high[i] - min_low[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral when no range
    
    # Get weekly trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA(21) for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=21, adjust=False).mean().values
    weekly_ema_1d = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(chop[i]) or np.isnan(weekly_ema_1d[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Alligator lines converge (teeth < lips) or weekly trend turns bearish
            if teeth[i] < lips[i] or close[i] < weekly_ema_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Alligator lines converge (teeth > lips) or weekly trend turns bullish
            if teeth[i] > lips[i] or close[i] > weekly_ema_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Only trade in trending markets (CHOP < 61.8) with volume spike
            if chop[i] < 61.8 and volume_spike[i]:
                # Enter long: Lips > Teeth > Jaw (bullish alignment) and price above weekly EMA
                if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > weekly_ema_1d[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short: Lips < Teeth < Jaw (bearish alignment) and price below weekly EMA
                elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < weekly_ema_1d[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals