#!/usr/bin/env python3
"""
Experiment #039: 6h Donchian(20) Breakout + 12h Trend + Volume Spike

HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts capture institutional 
participation when aligned with 12h trend (price > EMA50 for longs, < EMA50 for shorts) 
and confirmed by volume spike (>2.0x 20-period average). Uses discrete sizing 
(0.25) and ATR-based stoploss (2.5x ATR). Target: 80-180 trades over 4 years 
(20-45/year) with Sharpe > 0.5 on all symbols. Works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian20_12h_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours (00-23 UTC - trade all hours on 6h)
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 12h data for trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA(50) on 12h close
    if len(df_12h) >= 50:
        close_12h = df_12h['close'].values
        ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    else:
        ema_50_12h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume spike confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 6h Donchian Channel (20-period) ===
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    if n >= lookback:
        for i in range(lookback, n):
            highest_high[i] = np.max(high[i-lookback:i])
            lowest_low[i] = np.min(low[i-lookback:i])
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(100, lookback)  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in alignment with 12h EMA50 ---
        price_above_12h_ema = close[i] > ema_50_12h_aligned[i]
        price_below_12h_ema = close[i] < ema_50_12h_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long when price breaks above Donchian high with volume and trend alignment
        long_condition = (
            close[i] > highest_high[i] and 
            volume_spike and 
            price_above_12h_ema
        )
        
        # Short when price breaks below Donchian low with volume and trend alignment
        short_condition = (
            close[i] < lowest_low[i] and 
            volume_spike and 
            price_below_12h_ema
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>