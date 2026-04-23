#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Camarilla pivot levels (H3/L3) act as strong support/resistance on 4h chart
- Breakout above H3 with volume > 1.8x average signals bullish momentum
- Breakdown below L3 with volume > 1.8x average signals bearish momentum
- 1d EMA34 ensures trades align with daily trend (avoid counter-trend in choppy markets)
- Discrete position size 0.25 to minimize drawdown during crashes like 2022
- Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years)
- Uses tighter volume confirmation (1.8x) and smaller position (0.25) to reduce overtrading
- Optimized for BTC/ETH performance in both bull and bear regimes via 1d trend filter
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
    
    # Volume confirmation: > 1.8x 20-period average (tighter than v1's 2.0x)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for EMA34 trend filter (daily timeframe for stronger trend)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # volume MA, 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Close > H3 AND price above 1d EMA34 AND volume confirmation
            if close[i] > H3[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close < L3 AND price below 1d EMA34 AND volume confirmation
            elif close[i] < L3[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < pivot OR price crosses below 1d EMA34
            if close[i] < pivot[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > pivot OR price crosses above 1d EMA34
            if close[i] > pivot[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0