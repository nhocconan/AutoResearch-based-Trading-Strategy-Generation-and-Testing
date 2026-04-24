#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume spike confirmation.
- Uses Camarilla pivot levels (H3, L3) from prior completed 1w candles.
- Breakout above H3 or below L3 with volume > 2.0x 20-bar average signals strong momentum.
- Trend filter: price must be above/below 1w EMA50 to align with higher timeframe direction.
- Designed for 12h timeframe to capture medium-term breakouts in both bull and bear markets.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 12-37 trades/year (50-150 total over 4 years) to stay fee-efficient.
- Based on proven pattern: Camarilla breakout + volume + trend filter showed strong performance in DB.
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
    
    # Get 1w data ONCE before loop for Camarilla pivots and EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (H3, L3) from prior completed 1w candles
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Typical price for pivot calculation
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    # Camarilla width
    camarilla_width = (high_1w - low_1w) * 1.1 / 12.0
    # H3 and L3 levels
    h3_1w = close_1w + camarilla_width * 2.618
    l3_1w = close_1w - camarilla_width * 2.618
    
    # Align H3 and L3 to 12h timeframe (wait for 1w bar to close)
    h3_1w_aligned = align_htf_to_ltf(prices, df_1w, h3_1w)
    l3_1w_aligned = align_htf_to_ltf(prices, df_1w, l3_1w)
    
    # 1w EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3_1w_aligned[i]) or np.isnan(l3_1w_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above H3 AND price above 1w EMA50 AND volume confirmation
            if close[i] > h3_1w_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: breakout below L3 AND price below 1w EMA50 AND volume confirmation
            elif close[i] < l3_1w_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below L3 OR price below 1w EMA50
            if close[i] < l3_1w_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above H3 OR price above 1w EMA50
            if close[i] > h3_1w_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0