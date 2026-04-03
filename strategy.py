#!/usr/bin/env python3
"""
Experiment #1946: 4h Donchian Breakout + 1d Trend + Volume Confirmation
HYPOTHESIS: 4h Donchian(20) breakouts with 1d EMA(50) trend filter and volume spike (>1.5x) 
capture institutional breakouts in both bull and bear markets. 
Donchian channels provide objective price structure, while 1d trend ensures we trade 
with higher timeframe momentum. Volume confirmation filters false breakouts.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1946_4h_donchian20_1d_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian(20) and Volume MA(20) ===
    # Donchian channels: upper = max(high, 20), lower = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = 50  # sufficient for Donchian(20), EMA(50), volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
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
                tr = np.maximum(tr1, np.maximum(tr2, tr3))
                
                # Simple ATR calculation (we'll compute current ATR)
                atr_period = 14
                if i >= atr_period:
                    # Calculate ATR using Wilder's smoothing (equivalent to EMA with alpha=1/period)
                    tr_values = []
                    for j in range(i-atr_period+1, i+1):
                        tr1_j = high[j] - low[j]
                        tr2_j = abs(high[j] - close[j-1]) if j > 0 else tr1_j
                        tr3_j = abs(low[j] - close[j-1]) if j > 0 else tr1_j
                        tr_j = np.maximum(tr1_j, np.maximum(tr2_j, tr3_j))
                        tr_values.append(tr_j)
                    
                    if len(tr_values) >= atr_period:
                        atr = np.mean(tr_values)
                    else:
                        atr = np.mean(tr_values) if tr_values else 0.0
                else:
                    atr = 0.0
            else:
                atr = 0.0
            
            # Stoploss: exit if price moves 2*ATR against position
            exit_signal = False
            if position_side > 0:  # Long position
                if price < entry_price - 2.0 * atr:
                    exit_signal = True
            else:  # Short position
                if price > entry_price + 2.0 * atr:
                    exit_signal = True
            
            # Additional exit: Donchian opposite touch (mean reversion)
            if not exit_signal:
                if position_side > 0 and price <= donchian_lower[i]:
                    exit_signal = True
                elif position_side < 0 and price >= donchian_upper[i]:
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
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper AND 1d trend up
            if trend_1d_aligned[i] > 0 and price > donchian_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND 1d trend down
            elif trend_1d_aligned[i] < 0 and price < donchian_lower[i]:
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