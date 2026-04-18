#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1S1_VolumeSpike_Momentum
4h strategy using daily Camarilla pivot levels (R1/S1) with volume spike and momentum confirmation.
- Long: Close crosses above R1 + volume > 2x average + RSI > 50
- Short: Close crosses below S1 + volume > 2x average + RSI < 50
- Exit: Opposite crossover or RSI mean reversion (crosses 50)
Designed for ~20-40 trades/year per symbol (80-160 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    # Using formula: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are previous day's close, high, low
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day values (shift by 1 to avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day has no previous, set to NaN
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate Camarilla levels
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align daily levels to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # RSI(14) for momentum confirmation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough for volume MA and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 2.0 * vol_ma_aligned[i]
        
        # Price crossover conditions
        cross_above_R1 = close[i] > R1_aligned[i] and close[i-1] <= R1_aligned[i-1]
        cross_below_S1 = close[i] < S1_aligned[i] and close[i-1] >= S1_aligned[i-1]
        
        # RSI momentum
        rsi_bullish = rsi_values[i] > 50
        rsi_bearish = rsi_values[i] < 50
        
        if position == 0:
            # Long: cross above R1 + volume spike + RSI bullish
            if cross_above_R1 and vol_confirm and rsi_bullish:
                signals[i] = 0.25
                position = 1
            # Short: cross below S1 + volume spike + RSI bearish
            elif cross_below_S1 and vol_confirm and rsi_bearish:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: cross below S1 or RSI mean reversion (below 50)
            if cross_below_S1 or (rsi_values[i] < 50 and rsi_values[i-1] >= 50):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: cross above R1 or RSI mean reversion (above 50)
            if cross_above_R1 or (rsi_values[i] > 50 and rsi_values[i-1] <= 50):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1S1_VolumeSpike_Momentum"
timeframe = "4h"
leverage = 1.0