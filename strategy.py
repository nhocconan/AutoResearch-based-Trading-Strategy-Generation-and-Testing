#!/usr/bin/env python3
# 6h_OrderFlow_Imbalance_VWAP_Divergence
# Hypothesis: On 6h timeframe, price divergence from VWAP combined with volume imbalance
# (buying/selling pressure) signals mean-reversion in ranging markets and continuation
# in trending markets. Uses 1d trend filter to align with higher timeframe momentum.
# Designed for low trade frequency (~20-40/year) to minimize fee drag in bear markets.

name = "6h_OrderFlow_Imbalance_VWAP_Divergence"
timeframe = "6h"
leverage = 1.0

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
    
    # === 1d Trend Filter (Higher Timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 20-period EMA on daily for trend direction
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # === VWAP (6s) - Typical Price * Volume / Cumulative Volume ===
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    # Avoid division by zero
    vwap = np.where(cum_vol > 0, cum_pv / cum_vol, typical_price)
    
    # === Volume Imbalance (Buying vs Selling Pressure) ===
    # Using close location within the bar's range as proxy for aggression
    # Prevent division by zero in range
    ranges = high - low
    # Where range is zero, assume neutral (0.5)
    close_location = np.where(ranges > 0, (close - low) / ranges, 0.5)
    # Volume imbalance: positive = buying pressure, negative = selling pressure
    vol_imbalance = (2.0 * close_location - 1.0) * volume
    # Smooth with 3-period EMA to reduce noise
    vol_imbalance_smooth = pd.Series(vol_imbalance).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # === Price Deviation from VWAP (%) ===
    # Normalized deviation for signal scaling
    price_dev = (close - vwap) / (vwap + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure VWAP, EMA, and volume imbalance are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(vol_imbalance_smooth[i]) or np.isnan(price_dev[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions require alignment with 1d trend
        trend_up = close[i] > ema_20_1d_aligned[i]
        trend_down = close[i] < ema_20_1d_aligned[i]
        
        if position == 0:
            # LONG: Price significantly below VWAP (oversold) + buying pressure + uptrend bias
            if (price_dev[i] < -0.015 and  # 1.5% below VWAP
                vol_imbalance_smooth[i] > 0 and  # buying pressure
                trend_up):  # aligned with 1d uptrend
                signals[i] = 0.25
                position = 1
            # SHORT: Price significantly above VWAP (overbought) + selling pressure + downtrend bias
            elif (price_dev[i] > 0.015 and  # 1.5% above VWAP
                  vol_imbalance_smooth[i] < 0 and  # selling pressure
                  trend_down):  # aligned with 1d downtrend
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns to VWAP or trend changes
            if (price_dev[i] > -0.005 or  # back to near VWAP
                not trend_up):  # trend turned down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to VWAP or trend changes
            if (price_dev[i] < 0.005 or  # back to near VWAP
                not trend_down):  # trend turned up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals