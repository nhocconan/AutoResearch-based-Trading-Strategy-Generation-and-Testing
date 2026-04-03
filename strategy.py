#!/usr/bin/env python3
"""
Experiment #034: 1h Volume Spike + 4h/1d Trend Filter

HYPOTHESIS: On 1h timeframe, enter long when price breaks above recent high with volume spike (>2.0x average) 
and 4h/1d trend is bullish (price > 4h EMA20 and 1d EMA50). Enter short when price breaks below recent low 
with volume spike and 4h/1d trend is bearish. Use 4h/1d for signal direction, 1h only for entry timing precision. 
Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years (15-37/year) to minimize 
fee drag while capturing momentum bursts aligned with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_vol_spike_4h_1d_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Precompute session hours (08-20 UTC) ===
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    # === HTF: 4h data for trend and recent high/low ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate EMA(20) on 4h close
    if len(df_4h) >= 20:
        close_4h = df_4h['close'].values
        ema_20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
        ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    else:
        ema_20_4h_aligned = np.full(n, np.nan)
    
    # Calculate recent 4h high (20-bar) and low (20-bar) for breakout levels
    if len(df_4h) >= 20:
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
        low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
        high_20_4h_aligned = align_htf_to_ltf(prices, df_4h, high_20_4h)
        low_20_4h_aligned = align_htf_to_ltf(prices, df_4h, low_20_4h)
    else:
        high_20_4h_aligned = np.full(n, np.nan)
        low_20_4h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators ===
    # Volume ratio (current vs 20-period average)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_ratio = np.ones(n)  # Default to 1.0 for warmup period
        vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    else:
        vol_ratio = np.ones(n)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(high_20_4h_aligned[i]) or 
            np.isnan(low_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Alignment: 4h and 1d both bullish or both bearish ---
        bullish_4h = close[i] > ema_20_4h_aligned[i]
        bearish_4h = close[i] < ema_20_4h_aligned[i]
        bullish_1d = close[i] > ema_50_1d_aligned[i]
        bearish_1d = close[i] < ema_50_1d_aligned[i]
        
        both_bullish = bullish_4h and bullish_1d
        both_bearish = bearish_4h and bearish_1d
        
        # --- Volume Confirmation: Require significant spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
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
                # Take profit at 4h recent high/low (mean reversion to HTF range)
                if close[i] >= high_20_4h_aligned[i] or close[i] <= low_20_4h_aligned[i]:
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
                # Take profit at 4h recent high/low
                if close[i] >= high_20_4h_aligned[i] or close[i] <= low_20_4h_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Break above 4h recent high with volume spike in bullish 4h/1d
        long_condition = (
            both_bullish and 
            close[i] > high_20_4h_aligned[i] and 
            volume_spike
        )
        
        # Short: Break below 4h recent low with volume spike in bearish 4h/1d
        short_condition = (
            both_bearish and 
            close[i] < low_20_4h_aligned[i] and 
            volume_spike
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