#!/usr/bin/env python3
"""
Experiment #364: Daily Donchian Breakout + Weekly Trend Filter + Volume Spike

HYPOTHESIS: Daily Donchian(20) breakouts capture significant momentum moves. 
Filtering by weekly trend (price above/below weekly EMA50) ensures we trade with 
the higher timeframe direction. Volume confirmation (>1.5x 20-day average) 
confirms institutional participation. Uses discrete position sizing (0.25) 
and ATR-based stoploss (2.5x ATR) to manage risk. Targets 20-40 trades/year 
on 1d timeframe (80-160 total over 4 years) for minimal fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_1d_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: Weekly data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(50) on weekly close
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, close[-1] if n > 0 else 0.0)
    
    # === HTF: Weekly data for volume average (Call ONCE before loop) ===
    if len(df_1w) >= 20:
        volume_1w = df_1w['volume'].values
        vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
        vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    else:
        vol_ma_20_1w_aligned = np.full(n, 1.0)
    
    # === Daily Donchian(20) channels ===
    donchian_period = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        upper_channel[i] = np.max(high[i - donchian_period + 1:i + 1])
        lower_channel[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(100, donchian_period)  # Ensure enough data for HTF and Donchian
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Price above/below weekly EMA50 ---
        is_uptrend = close[i] > ema_50_1w_aligned[i]
        is_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # --- Volume Confirmation: Current daily volume > 1.5x weekly 20-period average ---
        # Note: Comparing daily volume to weekly average volume as proxy for participation
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_1w_aligned[i]
        
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
        # Long: Price breaks above upper Donchian channel in uptrend with volume
        long_condition = is_uptrend and volume_confirmed and (close[i] > upper_channel[i])
        
        # Short: Price breaks below lower Donchian channel in downtrend with volume
        short_condition = is_downtrend and volume_confirmed and (close[i] < lower_channel[i])
        
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