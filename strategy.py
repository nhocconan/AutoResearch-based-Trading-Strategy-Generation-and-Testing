#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation.
- Camarilla pivot levels (R3/S3) act as strong intraday support/resistance on 4h chart
- Breakout above R3 with volume > 2x average signals bullish momentum
- Breakdown below S3 with volume > 2x average signals bearish momentum
- 12h EMA50 ensures trades align with higher timeframe trend (avoid counter-trend)
- Discrete position size 0.30 to balance return and drawdown
- Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
- Works in both bull/bear via 12h trend filter and volatility-adjusted breakouts
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
    
    # Resistance levels
    R3 = pivot + (range_hl * 1.1 / 4.0)  # R3 = pivot + 1.1*(H-L)/4
    S3 = pivot - (range_hl * 1.1 / 4.0)  # S3 = pivot - 1.1*(H-L)/4
    
    # Volume confirmation: > 2.0x 20-period average
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
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close > R3 AND price above 12h EMA50 AND volume confirmation
            if close[i] > R3[i] and close[i] > ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = 0.30
                position = 1
            # Short: Close < S3 AND price below 12h EMA50 AND volume confirmation
            elif close[i] < S3[i] and close[i] < ema_50_12h_aligned[i] and volume_confirm:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: Close < pivot OR price crosses below 12h EMA50
            if close[i] < pivot[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: Close > pivot OR price crosses above 12h EMA50
            if close[i] > pivot[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0