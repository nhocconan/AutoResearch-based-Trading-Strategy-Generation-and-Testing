#!/usr/bin/env python3
"""
Experiment #4379: 6h Donchian Breakout + 12h Volume Spike + 1d ATR Regime Filter
HYPOTHESIS: Donchian(20) breakouts on 6h combined with 12h volume spikes (>2x MA) and filtered by 1d ATR regime (low volatility = range, high volatility = trend) capture institutional breakout moves while avoiding choppy markets. The ATR regime filter ensures we only trade when volatility is expanding (trending conditions), which improves breakout reliability. Volume confirmation adds conviction. Targets 50-150 total trades over 4 years (12-37/year) with position size 0.25. Works in bull via upward breakouts in high vol, in bear via downward breakouts in high vol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4379_6h_donchian20_12h_vol_1d_atr_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 12h volume for spike detection ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = align_htf_to_ltf(prices, df_12h, vol_12h / vol_ma_12h)
    else:
        vol_ratio_12h = np.full(n, np.nan)
    
    # === Precompute HTF: 1d ATR for regime filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        atr_ma_1d = pd.Series(atr_1d).rolling(window=30, min_periods=30).mean().values
        atr_ratio_1d = align_htf_to_ltf(prices, df_1d, atr_1d / atr_ma_1d)
    else:
        atr_ratio_1d = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_upper = high_series.rolling(window=20, min_periods=20).max().values
    donch_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 20, 30)  # Donchian, vol MA, ATR, vol ratio MA, atr ratio MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio_12h[i]) or
            np.isnan(atr_ratio_1d[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: 12h volume spike (> 2.0x average)
        volume_confirm = vol_ratio_12h[i] > 2.0
        
        # ATR regime filter: only trade when volatility is expanding (ratio > 1.2)
        vol_regime_confirm = atr_ratio_1d[i] > 1.2
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
        breakout_down = close[i] < donch_lower[i-1]  # Close below previous lower band
        
        # Long conditions: upward breakout + volume spike + volatility expansion
        long_entry = breakout_up and volume_confirm and vol_regime_confirm
        
        # Short conditions: downward breakout + volume spike + volatility expansion
        short_entry = breakout_down and volume_confirm and vol_regime_confirm
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals