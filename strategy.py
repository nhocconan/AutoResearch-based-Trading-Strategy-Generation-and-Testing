#!/usr/bin/env python3
"""
6h_1w_Momentum_Reversal_With_Volume
Hypothesis: Uses weekly momentum divergence and volume exhaustion to capture reversals in both bull and bear markets. 
The strategy looks for price making new highs/lows while momentum (RSI) fails to confirm, combined with declining volume 
to signal exhaustion. Works in ranging and trending markets by fading overextended moves. Targets 15-35 trades/year.
"""

name = "6h_1w_Momentum_Reversal_With_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly RSI for Momentum Divergence ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    rsi_1w = calculate_rsi(df_1w['close'].values, period=14)
    rsi_1w_6h = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # --- Weekly High/Low for Price Extremes ---
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_high_6h = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_6h = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # --- Volume Exhaustion (declining volume on new price extremes) ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # --- Price Position Relative to Weekly Range ---
    weekly_range = weekly_high_6h - weekly_low_6h
    weekly_range = np.where(weekly_range == 0, 1, weekly_range)  # avoid div by zero
    price_position = (close - weekly_low_6h) / weekly_range  # 0 = at weekly low, 1 = at weekly high
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_1w_6h[i]) or np.isnan(weekly_high_6h[i]) or 
            np.isnan(weekly_low_6h[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume exhaustion: volume declining on new extremes
        vol_exhaustion = vol_ratio[i] < 0.7  # volume below 70% of average
        
        if position == 0:
            # Bearish divergence: price at/near weekly high, RSI not confirming, volume exhausted
            if (price_position[i] > 0.85 and  # near weekly high
                rsi_1w_6h[i] < 60 and        # RSI not overbought (divergence)
                vol_exhaustion):
                signals[i] = -0.25
                position = -1
            # Bullish divergence: price at/near weekly low, RSI not confirming, volume exhausted
            elif (price_position[i] < 0.15 and   # near weekly low
                  rsi_1w_6h[i] > 40 and         # RSI not oversold (divergence)
                  vol_exhaustion):
                signals[i] = 0.25
                position = 1
        else:
            # Exit conditions: mean reversion back to middle of range or RSI normalization
            if position == 1:
                # Exit long: price returns to middle OR RSI becomes oversold
                if price_position[i] > 0.6 or rsi_1w_6h[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to middle OR RSI becomes overbought
                if price_position[i] < 0.4 or rsi_1w_6h[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals