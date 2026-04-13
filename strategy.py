# 6h_1w_1d_Camarilla_Pivot_With_Volume_and_Momentum_Filter_v1
# Hypothesis: Weekly Camarilla pivot levels combined with daily momentum and volume confirmation
# provides high-probability entries on 6h timeframe. Uses pivot structure for support/resistance,
# momentum to filter weak breakouts, and volume to confirm conviction. Designed to work in
# both bull and bear markets by focusing on mean reversion at extreme levels (R3/S3, R4/S4)
# and avoiding choppy periods via momentum filter.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete position sizing to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Daily data for momentum and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    # Using weekly high, low, close from previous completed week
    ph = df_1w['high'].values  # weekly high
    pl = df_1w['low'].values   # weekly low
    pc = df_1w['close'].values # weekly close
    
    # Camarilla equations
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    H_L = ph - pl
    camarilla_R4 = pc + (H_L * 1.1 / 2)
    camarilla_R3 = pc + (H_L * 1.1 / 4)
    camarilla_S3 = pc - (H_L * 1.1 / 4)
    camarilla_S4 = pc - (H_L * 1.1 / 2)
    
    # Align weekly Camarilla levels to 6h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_R4)
    R3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_S3)
    S4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_S4)
    
    # Daily momentum: RSI(14) to filter weak moves
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Daily volume confirmation: current 6h volume > 1.5x average 6h volume
    # 4 six-hour periods per day, so daily volume / 4 = approximate 6h period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_6h_avg_approx = volume_ma_20_1d / 4  # approximate 6h period MA
    volume_6h_avg_aligned = align_htf_to_ltf(prices, df_1d, volume_6h_avg_approx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(200, n):
        # Skip if any required data is not ready
        if (np.isnan(R4_aligned[i]) or np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(rsi_aligned[i]) or np.isnan(volume_6h_avg_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 6h volume > 1.5x average 6h volume
        volume_condition = volume[i] > (volume_6h_avg_aligned[i] * 1.5)
        
        # Momentum condition: RSI between 30 and 70 to avoid extreme readings
        rsi_momentum_ok = (rsi_aligned[i] > 30) and (rsi_aligned[i] < 70)
        
        # Fade logic at extreme levels (R3/S3, R4/S4)
        # Sell near R3/R4, buy near S3/S4
        fade_short = close[i] > R3_aligned[i] and volume_condition and rsi_momentum_ok
        fade_long = close[i] < S3_aligned[i] and volume_condition and rsi_momentum_ok
        
        # Breakout continuation at extreme levels (R4/S4)
        # Buy breakout above R4, sell breakdown below S4
        breakout_long = close[i] > R4_aligned[i] and volume_condition and rsi_momentum_ok
        breakout_short = close[i] < S4_aligned[i] and volume_condition and rsi_momentum_ok
        
        if position == 0:
            if fade_long:
                position = 1
                signals[i] = position_size
            elif fade_short:
                position = -1
                signals[i] = -position_size
            elif breakout_long:
                position = 1
                signals[i] = position_size
            elif breakout_short:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to mean (near weekly pivot) or stops at opposite extreme
            pivot = (R3_aligned[i] + S3_aligned[i]) / 2  # approximate midpoint
            if close[i] > pivot or close[i] < S3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to mean or stops at opposite extreme
            pivot = (R3_aligned[i] + S3_aligned[i]) / 2  # approximate midpoint
            if close[i] < pivot or close[i] > R3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_1d_Camarilla_Pivot_With_Volume_and_Momentum_Filter_v1"
timeframe = "6h"
leverage = 1.0