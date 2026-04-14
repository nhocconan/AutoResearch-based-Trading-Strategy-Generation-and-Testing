#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h momentum strategy using weekly RSI(14) + daily volume confirmation
# Weekly RSI provides smoothed momentum filter (avoids 6h noise) while daily volume confirms institutional participation
# Works in bull/bear: long when weekly RSI > 50 with volume surge, short when weekly RSI < 50 with volume surge
# Target: 50-150 total trades over 4 years (12-37/year) with selective entry conditions

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI(14)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1w = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    
    # Align weekly RSI to 6h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Get daily data for volume average
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate daily volume average (20-period)
    vol_1d_series = pd.Series(volume)
    avg_vol_1d = vol_1d_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Align daily volume average to 6h
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(avg_vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: weekly RSI > 50 (bullish momentum) with volume confirmation
            if (rsi_1w_aligned[i] > 50 and vol > 2.0 * avg_vol_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: weekly RSI < 50 (bearish momentum) with volume confirmation
            elif (rsi_1w_aligned[i] < 50 and vol > 2.0 * avg_vol_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: weekly RSI drops below 50
            if rsi_1w_aligned[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: weekly RSI rises above 50
            if rsi_1w_aligned[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Weekly_RSI_Volume_Momentum"
timeframe = "6h"
leverage = 1.0