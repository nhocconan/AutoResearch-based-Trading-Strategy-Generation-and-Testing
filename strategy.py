#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray power with 1d regime filter and volume confirmation
# Hypothesis: Elder Ray (bull/bear power) identifies trend strength; 1d trend filter avoids counter-trend trades; volume confirms momentum.
# Works in bull via strong bull power entries, in bear via bear power shorts with regime filter preventing whipsaws.
# Target: 15-35 trades/year to minimize fee drag.
name = "6h_elder_ray_1d_regime_volume_v2"
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
    
    # Get daily data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA22 on daily close for trend regime
    close_1d = df_1d['close'].values
    ema22_1d = pd.Series(close_1d).ewm(span=22, adjust=False, min_periods=22).mean().values
    ema22_1d_aligned = align_htf_to_ltf(prices, df_1d, ema22_1d)
    
    # Calculate 13-period EMA for Elder Ray (using 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 20-period moving average of volume for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(22, n):
        # Skip if required data not available
        if (np.isnan(ema22_1d_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 20-period average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Regime filter: 1d close above/below EMA22 determines bias
        bull_regime = close > ema22_1d_aligned[i]
        bear_regime = close < ema22_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: bear power becomes positive (selling pressure gone) OR regime turns bearish
            if bear_power[i] >= 0 or not bull_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: bull power becomes negative (buying pressure gone) OR regime turns bullish
            if bull_power[i] <= 0 or not bear_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: strong bull power + volume confirmation + bull regime
            if bull_power[i] > 0 and vol_confirm and bull_regime:
                position = 1
                signals[i] = 0.25
            # Enter short: strong bear power + volume confirmation + bear regime
            elif bear_power[i] < 0 and vol_confirm and bear_regime:
                position = -1
                signals[i] = -0.25
    
    return signals