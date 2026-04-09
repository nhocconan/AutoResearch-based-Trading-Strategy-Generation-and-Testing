#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels + volume confirmation + ATR stoploss
# Camarilla pivot levels from 1d provide strong support/resistance with proven edge on ETH/BTC
# Volume confirmation filters false breakouts (current 4h volume > 1.5x 20-period average)
# ATR trailing stop (2.5x ATR) manages risk and reduces whipsaw
# Designed for 4h timeframe targeting 20-50 trades/year (80-200 over 4 years)
# Works in bull/bear: price reacts to pivot levels, volume confirms validity, ATR stop adapts to volatility

name = "4h_1d_camarilla_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    pivot = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    
    # Camarilla levels: L3, L4, H3, H4
    camarilla_l3 = pivot - (1.1 * rng / 2)
    camarilla_l4 = pivot - (1.1 * rng)
    camarilla_h3 = pivot + (1.1 * rng / 2)
    camarilla_h4 = pivot + (1.1 * rng)
    
    # Align 1d Camarilla levels to 4h timeframe
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    
    # Pre-compute ATR(14) for 4h timeframe
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
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            if close[i] > highest_since_long:
                highest_since_long = close[i]
            # ATR trailing stop: exit if price drops 2.5x ATR from highest
            if close[i] < highest_since_long - 2.5 * atr[i]:
                position = 0
                highest_since_long = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if close[i] < lowest_since_short:
                lowest_since_short = close[i]
            # ATR trailing stop: exit if price rises 2.5x ATR from lowest
            if close[i] > lowest_since_short + 2.5 * atr[i]:
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