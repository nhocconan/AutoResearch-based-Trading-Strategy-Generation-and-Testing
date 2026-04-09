#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Camarilla pivot levels + volume confirmation + ATR trailing stop
# 1w Camarilla levels provide major weekly support/resistance with institutional relevance
# Volume confirmation requires daily volume > 1.5x 20-day average to filter noise
# ATR trailing stop (2.0x ATR) adapts to volatility and reduces whipsaw
# Designed for 1d timeframe targeting 10-25 trades/year (40-100 over 4 years)
# Works in bull/bear: price reacts to weekly pivot levels, volume confirms validity, ATR stop manages risk

name = "1d_1w_camarilla_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    pivot = (high_1w + low_1w + close_1w) / 3.0
    rng = high_1w - low_1w
    
    # Camarilla levels: L3, L4, H3, H4
    camarilla_l3 = pivot - (1.1 * rng / 2)
    camarilla_l4 = pivot - (1.1 * rng)
    camarilla_h3 = pivot + (1.1 * rng / 2)
    camarilla_h4 = pivot + (1.1 * rng)
    
    # Align 1w Camarilla levels to 1d timeframe
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    
    # Pre-compute ATR(14) for 1d timeframe
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average 1d volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            if close[i] > highest_since_long:
                highest_since_long = close[i]
            # ATR trailing stop: exit if price drops 2.0x ATR from highest
            if close[i] < highest_since_long - 2.0 * atr[i]:
                position = 0
                highest_since_long = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if close[i] < lowest_since_short:
                lowest_since_short = close[i]
            # ATR trailing stop: exit if price rises 2.0x ATR from lowest
            if close[i] > lowest_since_short + 2.0 * atr[i]:
                position = 0
                lowest_since_short = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion at Camarilla levels with volume confirmation
            # Long near L3/L4, Short near H3/H4
            if volume_confirmed:
                if close[i] <= camarilla_l3_aligned[i]:
                    position = 1
                    highest_since_long = close[i]
                    signals[i] = 0.25
                elif close[i] >= camarilla_h3_aligned[i]:
                    position = -1
                    lowest_since_short = close[i]
                    signals[i] = -0.25
    
    return signals