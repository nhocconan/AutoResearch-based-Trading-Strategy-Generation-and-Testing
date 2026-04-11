#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_breakout_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate Camarilla pivot levels from previous 1d bar (H, L, C)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels for previous day (H_{y-1}, L_{y-1}, C_{y-1})
    H_prev = np.roll(high_1d, 1)
    L_prev = np.roll(low_1d, 1)
    C_prev = np.roll(close_1d, 1)
    
    # First day: no previous data
    H_prev[0] = high_1d[0]
    L_prev[0] = low_1d[0]
    C_prev[0] = close_1d[0]
    
    # Calculate Camarilla levels
    range_prev = H_prev - L_prev
    camarilla_H4 = C_prev + range_prev * 1.1 / 2  # Resistance 4
    camarilla_L4 = C_prev - range_prev * 1.1 / 2  # Support 4
    camarilla_H3 = C_prev + range_prev * 1.1 / 4  # Resistance 3
    camarilla_L3 = C_prev - range_prev * 1.1 / 4  # Support 3
    camarilla_H2 = C_prev + range_prev * 1.1 / 6  # Resistance 2
    camarilla_L2 = C_prev - range_prev * 1.1 / 6  # Support 2
    camarilla_H1 = C_prev + range_prev * 1.1 / 12 # Resistance 1
    camarilla_L1 = C_prev - range_prev * 1.1 / 12 # Support 1
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    camarilla_H2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H2)
    camarilla_L2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L2)
    camarilla_H1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H1)
    camarilla_L1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L1)
    
    # Volume confirmation: volume > 1.5x 20-period average on 12h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(30, n):  # Start after 30 bars to ensure data availability
        # Skip if any required data is invalid
        if (np.isnan(vol_ma_20[i]) or np.isnan(camarilla_H4_aligned[i]) or 
            np.isnan(camarilla_L4_aligned[i]) or np.isnan(camarilla_H3_aligned[i]) or
            np.isnan(camarilla_L3_aligned[i]) or np.isnan(camarilla_H2_aligned[i]) or
            np.isnan(camarilla_L2_aligned[i]) or np.isnan(camarilla_H1_aligned[i]) or
            np.isnan(camarilla_L1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Breakout conditions
        # Long: price breaks above H4 (strong resistance) with volume
        long_signal = price_high > camarilla_H4_aligned[i] and volume_confirmed
        
        # Short: price breaks below L4 (strong support) with volume
        short_signal = price_low < camarilla_L4_aligned[i] and volume_confirmed
        
        # Exit conditions: return to H3/L3 levels (profit taking)
        exit_long = position == 1 and price_close < camarilla_H3_aligned[i]
        exit_short = position == -1 and price_close > camarilla_L3_aligned[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Camarilla pivot breakout strategy on 12h timeframe.
# Uses daily Camarilla levels (H4/L4) as breakout triggers with volume confirmation (>1.5x avg volume).
# Enters long when price breaks above H4 resistance, short when breaks below L4 support.
# Exits when price returns to H3/L3 levels for profit taking.
# Camarilla levels are based on previous day's high, low, close, providing statistical support/resistance.
# Works in both bull and bear markets by trading breakouts in either direction.
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drift.
# Uses 12h timeframe to capture significant moves while avoiding noise.