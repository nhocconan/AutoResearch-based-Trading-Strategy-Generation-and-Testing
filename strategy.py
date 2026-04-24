#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 12h EMA trend and volume confirmation.
- Primary timeframe: 6h targeting 80-160 total trades over 4 years (20-40/year).
- HTF: 12h for EMA50 trend filter (price above/below EMA50).
- Entry: Long when price breaks above H3 level AND 12h EMA50 uptrend AND volume > 1.5x average;
         Short when price breaks below L3 level AND 12h EMA50 downtrend AND volume > 1.5x average.
- Exit: Opposite breakout (price crosses H3/L3 in opposite direction) OR EMA trend reversal.
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla provides intraday support/resistance levels proven effective in crypto.
- Works in bull markets (buy H3 breakouts in uptrend) and bear markets (sell L3 breakdowns in downtrend).
- Estimated trades: ~120 total over 4 years (~30/year) based on H3/L3 breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the period."""
    # Typical price for the period
    typical_price = (high + low + close) / 3
    # Range
    range_val = high - low
    
    # Camarilla levels
    h5 = close + range_val * 1.1 / 2
    h4 = close + range_val * 1.1 / 4
    h3 = close + range_val * 1.1 / 6
    l3 = close - range_val * 1.1 / 6
    l4 = close - range_val * 1.1 / 4
    l5 = close - range_val * 1.1 / 2
    
    return h3, l3, h4, l4, h5, l5

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # EMA50 on 12h close
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Trend: price above EMA50 = uptrend, below = downtrend
    uptrend_12h = close_12h > ema_50_12h
    downtrend_12h = close_12h < ema_50_12h
    
    # Align 12h indicators to 6h timeframe
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h, additional_delay_bars=1)
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h, additional_delay_bars=1)
    
    # Calculate volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(uptrend_12h_aligned[i]) or np.isnan(downtrend_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels for current 6h bar
        h3, l3, _, _, _, _ = calculate_camarilla(high[i], low[i], close[i])
        
        curr_close = close[i]
        curr_volume = volume[i]
        vol_ratio = curr_volume / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation threshold
        volume_confirmed = vol_ratio > 1.5
        
        # Exit conditions: opposite breakout OR trend reversal
        if position != 0:
            # Exit long: price breaks below L3 OR downtrend
            if position == 1:
                if curr_close < l3 or downtrend_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above H3 OR uptrend
            elif position == -1:
                if curr_close > h3 or uptrend_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend and volume confirmation
        if position == 0:
            # Long: price breaks above H3 AND uptrend AND volume confirmed
            if curr_close > h3 and uptrend_12h_aligned[i] and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND downtrend AND volume confirmed
            elif curr_close < l3 and downtrend_12h_aligned[i] and volume_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_12hEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0