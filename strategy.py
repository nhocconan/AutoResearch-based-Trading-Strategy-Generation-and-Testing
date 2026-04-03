#!/usr/bin/env python3
"""
Experiment #029: 4h Donchian Breakout + Volume Spike + Choppiness Regime + 1d Trend Filter

HYPOTHESIS: Combines 4h Donchian channel breakouts with volume confirmation and 
choppiness regime filter to identify high-probability breakout trades. Uses 1d 
trend as higher timeframe filter to only trade breakouts in alignment with the 
daily trend. Donchian breakouts capture momentum, volume confirms institutional 
participation, and choppiness filter avoids false breakouts in ranging markets. 
Position size: 0.25 (25% of capital) with ATR-based stoploss. Target: 75-200 
trades over 4 years with tight entry conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_029_4h_donchian_vol_chop_1d_trend_v1"
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
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 4h Indicators: Donchian Channel (20) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2
    
    # === 4h Indicators: ATR(14) for volatility and stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume Spike (20-period volume MA) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)  # 50% above average volume
    
    # === 4h Indicators: Choppiness Index (14) for regime filter ===
    def choppiness_index(high, low, close, period=14):
        """Calculate Choppiness Index"""
        atr_sum = np.zeros(n)
        max_high = np.zeros(n)
        min_low = np.zeros(n)
        
        # Calculate True Range for each bar
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Sum of TR over period
        for i in range(period-1, n):
            atr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        for i in range(period-1, n):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        
        # Choppiness Index formula
        chop = np.full(n, np.nan)
        for i in range(period-1, n):
            if atr_sum[i] > 0 and max_high[i] != min_low[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Sufficient warmup for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- 1d Trend Filter: Only trade when price is clearly above/below EMA50 ---
        is_uptrend_1d = price > ema50_1d_aligned[i] * 1.001
        is_downtrend_1d = price < ema50_1d_aligned[i] * 0.999
        
        # --- Choppiness Regime Filter ---
        # CHOP > 61.8 = ranging market (avoid breakout trades)
        # CHOP < 38.2 = trending market (favor breakout trades)
        is_trending_market = chop[i] < 38.2
        is_ranging_market = chop[i] > 61.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_20[i-1]  # Break above previous period's high
        breakout_down = price < lowest_20[i-1]  # Break below previous period's low
        
        # --- Exit Logic (ATR-based trailing stop) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based trailing stop
            if position_side > 0:  # Long position
                # Trail stop: highest high since entry minus 2.5*ATR
                if bars_since_entry == 1:
                    highest_since_entry = high[i]
                else:
                    highest_since_entry = max(highest_since_entry, high[i])
                stop_level = highest_since_entry - 2.5 * atr_14[i]
                
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                    
            else:  # Short position
                # Trail stop: lowest low since entry plus 2.5*ATR
                if bars_since_entry == 1:
                    lowest_since_entry = low[i]
                else:
                    lowest_since_entry = min(lowest_since_entry, low[i])
                stop_level = lowest_since_entry + 2.5 * atr_14[i]
                
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Exit if market becomes ranging (choppiness too high)
            if is_ranging_market:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Only trade in trending markets with volume spike and aligned with 1d trend
        if is_trending_market and volume_spike[i]:
            # Long: Donchian breakout up AND 1d uptrend
            if breakout_up and is_uptrend_1d:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Donchian breakout down AND 1d downtrend
            elif breakout_down and is_downtrend_1d:
                in_position = True
                position_side = -1
                entry_price = close[i]
                lowest_since_entry = low[i]
                bars_since_entry = 0
                signals[i] = -SIZE
    
    return signals