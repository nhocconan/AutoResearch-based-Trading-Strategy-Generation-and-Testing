#!/usr/bin/env python3
"""
6h_OrderFlow_Imbalance_VWAP_Divergence_v2
Hypothesis: Detect institutional order flow imbalances via VWAP divergence on 6h timeframe, filtered by 1D trend and volume confirmation. 
Works in bull/bear by following 1D trend direction. VWAP divergence signals exhaustion of moves, allowing mean reversion entries.
Designed for low trade frequency (~20-40/year) to minimize fee drag while capturing high-probability reversals.
"""

name = "6h_OrderFlow_Imbalance_VWAP_Divergence_v2"
timeframe = "6h"
leverage = 1.0

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
    
    # === 1D Data for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1D EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === VWAP Calculation on 6h ===
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    # Avoid division by zero
    vwap = np.where(cum_vol > 0, cum_pv / cum_vol, typical_price)
    
    # === VWAP Divergence Detection ===
    # Bullish divergence: price makes lower low, VWAP makes higher low
    # Bearish divergence: price makes higher high, VWAP makes lower high
    # We'll use a 5-period lookback for simplicity
    lookback = 5
    
    # Price swings
    price_lower_low = np.zeros(n, dtype=bool)
    price_higher_high = np.zeros(n, dtype=bool)
    vwap_higher_low = np.zeros(n, dtype=bool)
    vwap_lower_high = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        # Price lower low: current low < lowest low in lookback period
        price_lower_low[i] = low[i] == np.min(low[i-lookback:i+1])
        # Price higher high: current high == highest high in lookback period
        price_higher_high[i] = high[i] == np.max(high[i-lookback:i+1])
        # VWAP higher low: current VWAP > lowest VWAP in lookback period
        vwap_higher_low[i] = vwap[i] > np.min(vwap[i-lookback:i+1])
        # VWAP lower high: current VWAP < highest VWAP in lookback period
        vwap_lower_high[i] = vwap[i] < np.max(vwap[i-lookback:i+1])
    
    # Volume filter: 1.5x 20-period EMA volume
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, lookback)
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vwap[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish VWAP divergence + uptrend on 1D + volume spike
            if (price_lower_low[i] and vwap_higher_low[i] and 
                close[i] > ema50_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish VWAP divergence + downtrend on 1D + volume spike
            elif (price_higher_high[i] and vwap_lower_high[i] and 
                  close[i] < ema50_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses above VWAP (momentum resumption) or trend fails
            if close[i] > vwap[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses below VWAP (momentum resumption) or trend fails
            if close[i] < vwap[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals