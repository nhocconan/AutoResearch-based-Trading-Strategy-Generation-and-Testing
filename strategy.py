#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d Camarilla pivot levels with volume confirmation and ATR trailing stop
# Uses 4h for signal direction (trend via Camarilla levels), 1d for stronger support/resistance
# Volume confirmation (current 1h volume > 2.0x 20-period average) filters false breakouts
# ATR trailing stop (2.5x ATR) manages risk and adapts to volatility
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# Target: 15-37 trades/year (60-150 over 4 years) to minimize fee drag
# Works in bull/bear: price reacts to multi-timeframe structure, volume confirms validity, ATR stop controls drawdown

name = "1h_4h_1d_camarilla_volume_atr_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) for filtering
    hours = prices.index.hour
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 25 or len(df_1d) < 25:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels (for trend direction)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    camarilla_r3_4h = close_4h + range_4h * 1.1 / 4.0
    camarilla_r4_4h = close_4h + range_4h * 1.1 / 2.0
    camarilla_s3_4h = close_4h - range_4h * 1.1 / 4.0
    camarilla_s4_4h = close_4h - range_4h * 1.1 / 2.0
    
    # Calculate 1d Camarilla pivot levels (stronger support/resistance)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_r3_1d = close_1d + range_1d * 1.1 / 4.0
    camarilla_r4_1d = close_1d + range_1d * 1.1 / 2.0
    camarilla_s3_1d = close_1d - range_1d * 1.1 / 4.0
    camarilla_s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels to 1h timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    r4_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    s4_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4_4h)
    
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    
    # Pre-compute ATR(14) for 1h timeframe
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
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(r4_4h_aligned[i]) or
            np.isnan(s3_4h_aligned[i]) or np.isnan(s4_4h_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 2.0x average 1h volume
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
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
                signals[i] = 0.20
                
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
                signals[i] = -0.20
        else:  # Flat
            # Breakout trading with volume confirmation and multi-timeframe confluence
            # Require both 4h and 1d levels to align for stronger signal
            if volume_confirmed:
                # Long when price breaks above both 4h R4 and 1d R4
                if close[i] > r4_4h_aligned[i] and close[i] > r4_1d_aligned[i]:
                    position = 1
                    highest_since_long = close[i]
                    signals[i] = 0.20
                # Short when price breaks below both 4h S4 and 1d S4
                elif close[i] < s4_4h_aligned[i] and close[i] < s4_1d_aligned[i]:
                    position = -1
                    lowest_since_short = close[i]
                    signals[i] = -0.20
    
    return signals