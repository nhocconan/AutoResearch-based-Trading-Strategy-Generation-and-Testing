#!/usr/bin/env python3
"""
Experiment #356: 12h Donchian Breakout + 1d Volume Confirmation + ATR Stoploss

HYPOTHESIS: Donchian(20) breakouts on 12h timeframe, confirmed by 1d volume spike (>2x average) 
and filtered by 1d trend (price > EMA200 for long, < EMA200 for short), captures strong 
momentum moves while minimizing false breakouts. The 12h timeframe targets 12-37 trades/year 
(50-150 total over 4 years) to reduce fee drag. Volume confirmation ensures institutional 
participation, and the EMA200 filter aligns with higher timeframe direction. ATR-based 
stoploss (2.5x) manages risk. Discrete position sizing (0.25) minimizes churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_trend_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume confirmation and trend filter (Call ONCE before loop) ===
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
    
    # Calculate EMA(200) on 1d close for trend filter
    if len(df_1d) >= 200:
        close_1d = df_1d['close'].values
        ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    else:
        ema_200_1d_aligned = np.full(n, np.nan)
    
    # === 12h Indicators ===
    # Calculate Donchian(20) channels on 12h
    # Need to get 12h OHLC data
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) >= 20:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        
        # Calculate rolling max/min for Donchian channels
        donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
        
        # Align to LTF (12h -> 12h is identity, but we still use align_htf_to_ltf for consistency)
        donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
        donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    else:
        donchian_high_aligned = np.full(n, np.nan)
        donchian_low_aligned = np.full(n, np.nan)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 200  # Ensure enough data for EMA200 and Donchian calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in direction of 1d EMA200 ---
        price_above_ema = close[i] > ema_200_1d_aligned[i]
        price_below_ema = close[i] < ema_200_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss using LTF data
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
        # Long: Price breaks above Donchian high with volume confirmation and uptrend
        long_condition = (
            close[i] > donchian_high_aligned[i] and 
            volume_spike and 
            price_above_ema
        )
        
        # Short: Price breaks below Donchian low with volume confirmation and downtrend
        short_condition = (
            close[i] < donchian_low_aligned[i] and 
            volume_spike and 
            price_below_ema
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