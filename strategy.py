#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation.
- Uses 1h timeframe for entry timing precision, 4h for signal direction (trend filter)
- Camarilla H3/L3 breakouts on 1h with volume > 1.5x average capture momentum
- 4h EMA50 ensures trades align with intermediate trend (avoid counter-trend in chop)
- Session filter (08-20 UTC) reduces noise trades during low-liquidity hours
- Discrete position size 0.20 to minimize drawdown and fee churn
- Target: 15-30 trades/year on 1h (60-120 total over 4 years)
- Volume confirmation tightened to 1.5x to reduce overtrading vs typical 2.0x
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
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate typical price for Camarilla pivots (using prior bar's OHLC)
    typical_price = (high + low + close) / 3.0
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
    H3 = pivot + (range_hl * 1.1 / 6.0)
    L3 = pivot - (range_hl * 1.1 / 6.0)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # volume MA, 4h EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Close > H3 AND price above 4h EMA50 AND volume confirmation
            if close[i] > H3[i] and close[i] > ema_50_4h_aligned[i] and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short: Close < L3 AND price below 4h EMA50 AND volume confirmation
            elif close[i] < L3[i] and close[i] < ema_50_4h_aligned[i] and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Close < pivot OR price crosses below 4h EMA50
            if close[i] < pivot[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Close > pivot OR price crosses above 4h EMA50
            if close[i] > pivot[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0