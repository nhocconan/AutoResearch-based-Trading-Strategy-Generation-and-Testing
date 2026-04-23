#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA50 trend filter and volume confirmation.
- Uses 12h timeframe for HTF trend (more responsive than 1d for 4h strategy)
- Volume confirmation > 1.6x average to reduce false breakouts
- Discrete position size 0.25 to balance risk and reward
- Target: 25-35 trades/year on 4h timeframe (100-140 total over 4 years)
- Focuses on BTC/ETH performance with proven Camarilla structure
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate typical price for Camarilla pivots (using prior bar's OHLC)
    typical_price = (high + low + close) / 3.0
    
    # Shift by 1 to use prior bar's data for pivot calculation (no look-ahead)
    typical_price_shifted = np.roll(typical_price, 1)
    high_shifted = np.roll(high, 1)
    low_shifted = np.roll(low, 1)
    close_shifted = np.roll(close, 1)
    
    # Set first bar to NaN since we don't have prior bar data
    typical_price_shifted[0] = np.nan
    high_shifted[0] = np.nan
    low_shifted[0] = np.nan
    close_shifted[0] = np.nan
    
    # Camarilla pivot levels (based on prior bar)
    pivot = (high_shifted + low_shifted + close_shifted) / 3.0
    range_hl = high_shifted - low_shifted
    
    # Resistance/support levels (H3/L3 = 1.1*(H-L)/6 from pivot)
    H3 = pivot + (range_hl * 1.1 / 6.0)  # H3 = pivot + 1.1*(H-L)/6
    L3 = pivot - (range_hl * 1.1 / 6.0)  # L3 = pivot - 1.1*(H-L)/6
    
    # Volume confirmation: > 1.6x 20-period average (balanced for trade frequency)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # volume MA, 12h EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.6x average)
        volume_confirm = volume[i] > 1.6 * vol_ma[i]
        
        if position == 0:
            # Long: Close > H3 AND price above 12h EMA50 AND volume confirmation
            if close[i] > H3[i] and close[i] > ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close < L3 AND price below 12h EMA50 AND volume confirmation
            elif close[i] < L3[i] and close[i] < ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < pivot OR price crosses below 12h EMA50
            if close[i] < pivot[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > pivot OR price crosses above 12h EMA50
            if close[i] > pivot[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_12hEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0