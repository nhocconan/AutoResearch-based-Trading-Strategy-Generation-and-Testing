#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H4/L4 breakout with 1w EMA34 trend filter and volume spike confirmation.
- Camarilla pivot levels (H4/L4) act as strong support/resistance on 12h chart
- Breakout above H4 with volume > 2.0x average signals bullish momentum
- Breakdown below L4 with volume > 2.0x average signals bearish momentum
- 1w EMA34 ensures trades align with higher timeframe trend (avoid counter-trend)
- Discrete position size 0.25 to balance return and drawdown
- Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
- Works in both bull/bear via 1w trend filter and volatility-adjusted breakouts
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
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # volume MA, 1w EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(H4[i]) or np.isnan(L4[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close > H4 AND price above 1w EMA34 AND volume confirmation
            if close[i] > H4[i] and close[i] > ema_34_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close < L4 AND price below 1w EMA34 AND volume confirmation
            elif close[i] < L4[i] and close[i] < ema_34_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < pivot OR price crosses below 1w EMA34
            if close[i] < pivot[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > pivot OR price crosses above 1w EMA34
            if close[i] > pivot[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H4L4_Breakout_1wEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0