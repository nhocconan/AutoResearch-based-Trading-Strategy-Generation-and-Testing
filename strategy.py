#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H4/L4 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Camarilla pivot levels (H4/L4) act as strong support/resistance on 12h chart
- Breakout above H4 with volume > 1.8x average signals bullish momentum
- Breakdown below L4 with volume > 1.8x average signals bearish momentum
- 1d EMA34 ensures trades align with higher timeframe trend (avoid counter-trend)
- Discrete position size 0.25 to minimize fee churn while maintaining profitability
- Target: 12-30 trades/year on 12h timeframe (50-120 total over 4 years)
- Works in both bull/bear via 1d trend filter and volatility-adjusted breakouts
- Designed for low trade frequency to overcome fee drag in ranging/bear markets
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
    
    # H4 and L4 levels (Camarilla)
    H4 = pivot + (range_hl * 1.1 / 2.0)  # H4 = pivot + 1.1*(H-L)/2
    L4 = pivot - (range_hl * 1.1 / 2.0)  # L4 = pivot - 1.1*(H-L)/2
    
    # Volume confirmation: > 1.8x 24-period average (more lenient for 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 34)  # volume MA, 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(H4[i]) or np.isnan(L4[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Close > H4 AND price above 1d EMA34 AND volume confirmation
            if close[i] > H4[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close < L4 AND price below 1d EMA34 AND volume confirmation
            elif close[i] < L4[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm:
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

name = "12h_Camarilla_H4L4_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0