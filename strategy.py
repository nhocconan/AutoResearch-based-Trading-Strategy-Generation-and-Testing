#!/usr/bin/env python3
"""
Experiment #1922: 12h Donchian(20) Breakout + 1d EMA Trend + Volume Confirmation
HYPOTHESIS: 12h Donchian(20) breakouts capture medium-term trends. 
Filter with 1d EMA(50) trend and volume spike (>1.5x 20-period average) to avoid false breakouts.
Works in bull markets by riding uptrends and bear markets by shorting downtrends.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1922_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 12h Indicators: Donchian(20) channels ===
    # Donchian upper = max(high, lookback=20)
    # Donchian lower = min(low, lookback=20)
    lookback = 20
    dc_upper = np.full(n, np.nan)
    dc_lower = np.full(n, np.nan)
    
    for i in range(lookback, n):
        dc_upper[i] = np.max(high[i-lookback:i])
        dc_lower[i] = np.min(low[i-lookback:i])
    
    # Volume MA(20) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20) and EMA(50)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss (2*ATR) ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR(14) for dynamic stoploss
            if i >= 14:
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr = max(tr1, tr2, tr3)
                # Simple ATR approximation using recent TR
                atr_approx = np.mean([
                    max(high[i-13] - low[i-13], abs(high[i-13] - close[i-14]), abs(low[i-13] - close[i-14])),
                    max(high[i-12] - low[i-12], abs(high[i-12] - close[i-13]), abs(low[i-12] - close[i-13])),
                    max(high[i-11] - low[i-11], abs(high[i-11] - close[i-12]), abs(low[i-11] - close[i-12])),
                    max(high[i-10] - low[i-10], abs(high[i-10] - close[i-11]), abs(low[i-10] - close[i-11])),
                    max(high[i-9] - low[i-9], abs(high[i-9] - close[i-10]), abs(low[i-9] - close[i-10])),
                    max(high[i-8] - low[i-8], abs(high[i-8] - close[i-9]), abs(low[i-8] - close[i-9])),
                    max(high[i-7] - low[i-7], abs(high[i-7] - close[i-8]), abs(low[i-7] - close[i-8])),
                    max(high[i-6] - low[i-6], abs(high[i-6] - close[i-7]), abs(low[i-6] - close[i-7])),
                    max(high[i-5] - low[i-5], abs(high[i-5] - close[i-6]), abs(low[i-5] - close[i-6])),
                    max(high[i-4] - low[i-4], abs(high[i-4] - close[i-5]), abs(low[i-4] - close[i-5])),
                    max(high[i-3] - low[i-3], abs(high[i-3] - close[i-4]), abs(low[i-3] - close[i-4])),
                    max(high[i-2] - low[i-2], abs(high[i-2] - close[i-3]), abs(low[i-2] - close[i-3])),
                    max(high[i-1] - low[i-1], abs(high[i-1] - close[i-2]), abs(low[i-1] - close[i-2])),
                    tr
                ]) / 14
            else:
                atr_approx = 0.0
            
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Stoploss: price drops 2*ATR below entry
                if price < entry_price - 2.0 * atr_approx:
                    exit_signal = True
                # Optional: take profit at 3*ATR profit
                elif price > entry_price + 3.0 * atr_approx:
                    exit_signal = True
            else:  # Short position
                # Stoploss: price rises 2*ATR above entry
                if price > entry_price + 2.0 * atr_approx:
                    exit_signal = True
                # Optional: take profit at 3*ATR profit
                elif price < entry_price - 3.0 * atr_approx:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias filter
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper AND 1d trend up
            if trend_bias > 0 and price > dc_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND 1d trend down
            elif trend_bias < 0 and price < dc_lower[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals