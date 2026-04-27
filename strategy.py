#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# ==========================================================
# Strategy: 6h_RVI_Period_7_TF_Signal_1d
# Hypothesis: 6h Relative Vigor Index (RVI) with 1d EMA trend filter.
# - RVI(7) measures conviction of price moves; values > 0.5 indicate bullish momentum, < -0.5 bearish.
# - Uses 1d EMA(34) as trend filter: only long when price > EMA, short when price < EMA.
# - Designed for low frequency (~20-30 trades/year) to minimize fee drag on 6s timeframe.
# - Works in bull/bear: RVI adapts to momentum shifts, EMA filter avoids counter-trend trades.
# ==========================================================

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34)
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2 / 35) + (ema_34_1d[i-1] * 33 / 35)
    
    # Align 1d EMA to 6h (no extra delay needed for EMA)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate RVI(7) on 6s data
    # RVI = [SMMA((Close - Open) + 2*(Close-1 - Open-1) + 2*(Close-2 - Open-2) + (Close-3 - Open-3))] /
    #       [SMMA((High - Low) + 2*(High-1 - Low-1) + 2*(High-2 - Low-2) + (High-3 - Low-3))]
    # where SMMA is smoothed moving average (same as Wilder's smoothing)
    
    num = np.zeros(n)  # numerator: close - open weighted
    den = np.zeros(n)  # denominator: high - low weighted
    
    # Calculate weighted components
    a = (close - open_price) + 2 * np.roll(close - open_price, 1) + 2 * np.roll(close - open_price, 2) + np.roll(close - open_price, 3)
    b = (high - low) + 2 * np.roll(high - low, 1) + 2 * np.roll(high - low, 2) + np.roll(high - low, 3)
    
    # Handle roll NaNs for first 3 elements
    a[:3] = 0
    b[:3] = 0
    
    # Wilder's smoothing (SMMA) with period 7
    def wilder_smooth(x, period):
        result = np.full_like(x, np.nan)
        if len(x) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(x[:period])
        # Subsequent values: smoothed = (prev * (period-1) + current) / period
        for i in range(period, len(x)):
            if not np.isnan(result[i-1]) and not np.isnan(x[i]):
                result[i] = (result[i-1] * (period-1) + x[i]) / period
            else:
                result[i] = np.nan
        return result
    
    num_smooth = wilder_smooth(a, 7)
    den_smooth = wilder_smooth(b, 7)
    
    # RVI = num_smooth / den_smooth, handle division by zero
    rvi = np.full(n, np.nan)
    mask = den_smooth != 0
    rvi[mask] = num_smooth[mask] / den_smooth[mask]
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup: need enough for RVI smoothing and EMA
    start_idx = max(34, 10)  # EMA needs 34, RVI needs ~10 for smoothing
    
    for i in range(start_idx, n):
        if np.isnan(rvi[i]) or np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: RVI > 0.5 (bullish momentum) AND price above 1d EMA
            if rvi[i] > 0.5 and price > ema_34_1d_aligned[i]:
                signals[i] = size
                position = 1
            # Short: RVI < -0.5 (bearish momentum) AND price below 1d EMA
            elif rvi[i] < -0.5 and price < ema_34_1d_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RVI < 0 (momentum lost) OR price crosses below EMA
            if rvi[i] < 0 or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: RVI > 0 (momentum lost) OR price crosses above EMA
            if rvi[i] > 0 or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RVI_Period_7_TF_Signal_1d"
timeframe = "6h"
leverage = 1.0
# ==========================================================
# End of strategy
# ==========================================================