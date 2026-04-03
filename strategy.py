#!/usr/bin/env python3
"""
Experiment #1974: 1h Donchian Breakout + 4h/1d Trend + Volume Confirmation
HYPOTHESIS: 1h Donchian(20) breakouts aligned with 4h EMA(20) trend and 1d EMA(50) filter,
with volume confirmation (>1.5x 20-period average) and session filter (08-20 UTC),
capture institutional breakouts in both bull and bear markets. Target: 60-150 total trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1974_1h_donchian20_4h_1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for EMA(20) trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    trend_4h = np.where(close_4h > ema_20_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # === HTF: 1d data for EMA(50) trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 1h Indicators: Donchian(20) channels ===
    donchian_window = 20
    upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Session filter: 08-20 UTC (pre-compute hours array) ===
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(donchian_window, 20, 50)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss (2*ATR) and time-based exit ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR(14) for dynamic stoploss
            if i >= 14:
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr = max(tr1, tr2, tr3)
                # Simple ATR calculation using previous value
                if i == 14:
                    atr = tr
                else:
                    atr = 0.93 * atr_prev + 0.07 * tr  # Wilder's smoothing
                atr_prev = atr
            else:
                atr = 1.0  # fallback
            
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Stoploss: 2*ATR below entry
                if price < entry_price - 2.0 * atr:
                    exit_signal = True
                # Time exit: close position after 24 bars (1 day)
                elif bars_since_entry >= 24:
                    exit_signal = True
                # Reverse signal: opposite Donchian break
                elif price < lower[i]:
                    exit_signal = True
            else:  # Short position
                # Stoploss: 2*ATR above entry
                if price > entry_price + 2.0 * atr:
                    exit_signal = True
                # Time exit: close position after 24 bars (1 day)
                elif bars_since_entry >= 24:
                    exit_signal = True
                # Reverse signal: opposite Donchian break
                elif price > upper[i]:
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
        # Require both 4h and 1d trend alignment for bias filter
        trend_bias_4h = trend_4h_aligned[i]
        trend_bias_1d = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike and trend_bias_4h == trend_bias_1d:
            # Long entry: price breaks above upper Donchian AND both trends up
            if trend_bias_4h > 0 and trend_bias_1d > 0 and price > upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND both trends down
            elif trend_bias_4h < 0 and trend_bias_1d < 0 and price < lower[i]:
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