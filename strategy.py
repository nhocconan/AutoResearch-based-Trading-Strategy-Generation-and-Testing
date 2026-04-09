#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d/1w Camarilla pivot levels with volume confirmation and ATR trailing stop
# Camarilla pivots from 1d/1w provide multi-timeframe support/resistance levels
# Volume confirmation (current 6h volume > 1.8x 20-period average) filters false breakouts
# ATR trailing stop (2.2x ATR) manages risk and adapts to volatility
# Designed for 6h timeframe targeting 15-25 trades/year (60-100 over 4 years)
# Works in bull/bear: price reacts to 1d/1w structure, volume confirms validity, ATR stop controls drawdown

name = "6h_1d_1w_camarilla_volume_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 25 or len(df_1w) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla pivot levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_r3_1d = close_1d + range_1d * 1.1 / 4.0
    camarilla_r4_1d = close_1d + range_1d * 1.1 / 2.0
    camarilla_s3_1d = close_1d - range_1d * 1.1 / 4.0
    camarilla_s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Calculate 1w Camarilla pivot levels (stronger levels)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    camarilla_r3_1w = close_1w + range_1w * 1.1 / 4.0
    camarilla_r4_1w = close_1w + range_1w * 1.1 / 2.0
    camarilla_s3_1w = close_1w - range_1w * 1.1 / 4.0
    camarilla_s4_1w = close_1w - range_1w * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w)
    
    # Pre-compute ATR(14) for 6h timeframe
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
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(r3_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.8x average 6h volume
        volume_confirmed = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            if close[i] > highest_since_long:
                highest_since_long = close[i]
            # ATR trailing stop: exit if price drops 2.2x ATR from highest
            if close[i] < highest_since_long - 2.2 * atr[i]:
                position = 0
                highest_since_long = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if close[i] < lowest_since_short:
                lowest_since_short = close[i]
            # ATR trailing stop: exit if price rises 2.2x ATR from lowest
            if close[i] > lowest_since_short + 2.2 * atr[i]:
                position = 0
                lowest_since_short = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout trading with volume confirmation and HTF alignment
            # Long on Camarilla R4 breakout (both 1d and 1w aligned)
            # Short on Camarilla S4 breakout (both 1d and 1w aligned)
            if volume_confirmed:
                # Require both 1d and 1w R4/S4 levels to be broken for stronger signal
                if close[i] > r4_1d_aligned[i] and close[i] > r4_1w_aligned[i]:
                    position = 1
                    highest_since_long = close[i]
                    signals[i] = 0.25
                elif close[i] < s4_1d_aligned[i] and close[i] < s4_1w_aligned[i]:
                    position = -1
                    lowest_since_short = close[i]
                    signals[i] = -0.25
    
    return signals