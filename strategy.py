#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1d EMA50 trend filter and volume confirmation.
# Camarilla levels derived from previous day's range provide institutional support/resistance.
# Trade breakouts above R1 (bullish) or below S1 (bearish) only when aligned with 1d EMA50 trend.
# Volume confirmation ensures institutional participation. Designed for 12h timeframe to limit trades.
# Works in both bull and bear markets by following higher timeframe trend and avoiding counter-trend trades.
# Target: 15-30 trades per year to minimize fee drag.

name = "12h_Camarilla_R1S1_Breakout_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for EMA50 trend and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d EMA50 for trend direction ===
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === 1d OHLC for Camarilla pivot calculation (using previous day's data) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels for each day based on previous day's OHLC
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    camarilla_r1 = close_1d_vals + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d_vals - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 12h Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema_50_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price closes above R1 with uptrend and volume confirmation
            if close_val > r1_val and ema_val > 0 and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below S1 with downtrend and volume confirmation
            elif close_val < s1_val and ema_val < 0 and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below S1 (reversal) or trend change
            if close_val < s1_val or ema_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above R1 (reversal) or trend change
            if close_val > r1_val or ema_val >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals