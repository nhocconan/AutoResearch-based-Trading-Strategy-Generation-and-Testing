#!/usr/bin/env python3
"""
Experiment #1979: 6h Donchian(20) Breakout + 12h Trend Filter + Volume Spike
HYPOTHESIS: Donchian channel breakouts on 6h timeframe capture medium-term momentum. 
Filtering by 12h EMA(50) trend ensures alignment with higher timeframe direction, 
while volume spike (>2x 20-period average) confirms institutional participation. 
This combination should work in both bull and bear markets by following the 
established trend with volume confirmation, reducing false breakouts.
Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1979_6h_donchian20_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h EMA(50) trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_12h = np.where(close_12h > ema_50_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === 6h Indicators: Donchian(20) channels ===
    # Donchian upper = max(high, lookback=20)
    # Donchian lower = min(low, lookback=20)
    lookback = 20
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    for i in range(lookback, n):
        donchian_upper[i] = np.max(high[i-lookback:i])
        donchian_lower[i] = np.min(low[i-lookback:i])
    
    # 6h Volume MA(20) for spike detection
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
    
    warmup = max(50, lookback)  # sufficient for EMA(50) and Donchian(20)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based trailing stop (2*ATR) ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR(14) for dynamic stoploss
            if i >= 14:
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr = max(tr1, tr2, tr3)
                
                # Calculate ATR using Wilder's smoothing
                if i == 14:
                    atr = np.mean([max(high[j]-low[j], abs(high[j]-close[j-1]), abs(low[j]-close[j-1])) 
                                 for j in range(i-13, i+1)])
                else:
                    atr = (atr_prev * 13 + tr) / 14
            else:
                atr = 0.0
            
            atr_prev = atr  # store for next iteration
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Trailing stop: exit if price drops 2*ATR from highest high since entry
                if bars_since_entry == 1:
                    entry_high = high[i]
                else:
                    entry_high = max(entry_high, high[i])
                
                if price < entry_high - 2.0 * atr:
                    exit_signal = True
            else:  # Short position
                # Trailing stop: exit if price rises 2*ATR from lowest low since entry
                if bars_since_entry == 1:
                    entry_low = low[i]
                else:
                    entry_low = min(entry_low, low[i])
                
                if price > entry_low + 2.0 * atr:
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
        # Require 12h trend alignment for bias filter
        trend_bias = trend_12h_aligned[i]
        
        # Volume confirmation: require volume spike (> 2x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper AND 12h trend up
            if trend_bias > 0 and price > donchian_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_high = high[i]  # initialize trailing stop high
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND 12h trend down
            elif trend_bias < 0 and price < donchian_lower[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_low = low[i]  # initialize trailing stop low
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals