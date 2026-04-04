#!/usr/bin/env python3
"""
Experiment #3519: 6h Donchian(20) Breakout + 12h Supertrend Filter + Volume Confirmation
HYPOTHESIS: 6h Donchian breakouts with 12h Supertrend direction filter and volume confirmation capture medium-term trends while avoiding whipsaws. 
Supertrend on 12h provides robust trend direction (works in bull/bear via ATR-based adaptive stop). Volume confirms breakout authenticity. 
Position size 0.25. Target: 100-200 total trades over 4 years (25-50/year).
Uses 12h for trend filter, 6h only for entry timing and risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3519_6h_donchian20_12h_supertrend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 12h data for Supertrend trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Supertrend (12h, ATR=10, multiplier=3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr_12h).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl2_12h = (high_12h + low_12h) / 2.0
    upper_basic_12h = hl2_12h + multiplier * atr_12h
    lower_basic_12h = hl2_12h - multiplier * atr_12h
    
    # Final Upper and Lower Bands
    upper_final_12h = np.full_like(close_12h, np.nan)
    lower_final_12h = np.full_like(close_12h, np.nan)
    
    for i in range(1, len(close_12h)):
        # Upper Band
        if close_12h[i-1] <= upper_final_12h[i-1]:
            upper_final_12h[i] = min(upper_basic_12h[i], upper_final_12h[i-1])
        else:
            upper_final_12h[i] = upper_basic_12h[i]
        
        # Lower Band
        if close_12h[i-1] >= lower_final_12h[i-1]:
            lower_final_12h[i] = max(lower_basic_12h[i], lower_final_12h[i-1])
        else:
            lower_final_12h[i] = lower_basic_12h[i]
    
    # Supertrend Direction
    supertrend_12h = np.full_like(close_12h, np.nan)
    for i in range(len(close_12h)):
        if np.isnan(upper_final_12h[i]) or np.isnan(lower_final_12h[i]):
            supertrend_12h[i] = np.nan
        elif close_12h[i] <= upper_final_12h[i]:
            supertrend_12h[i] = upper_final_12h[i]
        else:
            supertrend_12h[i] = lower_final_12h[i]
    
    # Trend: 1 = uptrend (price > supertrend), -1 = downtrend (price < supertrend)
    trend_12h = np.where(close_12h > supertrend_12h, 1, -1)
    trend_12h = np.where(np.isnan(supertrend_12h), 0, trend_12h)
    
    # Align Supertrend trend to 6h timeframe
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === 6h Indicators: Donchian channels (20-period) for entry timing ===
    lookback_6h = 20
    highest_high_6h = pd.Series(high).rolling(window=lookback_6h, min_periods=lookback_6h).max().values
    lowest_low_6h = pd.Series(low).rolling(window=lookback_6h, min_periods=lookback_6h).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_6h, atr_period + 1, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high_6h[i]) or np.isnan(lowest_low_6h[i]) or
            np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if trend changes to bearish
                elif trend_12h_aligned[i] < 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if trend changes to bullish
                elif trend_12h_aligned[i] > 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) for confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Long entry: price breaks above 6h Donchian high with bullish 12h trend
            if (price > highest_high_6h[i] and 
                trend_12h_aligned[i] > 0):  # Bullish 12h trend
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below 6h Donchian low with bearish 12h trend
            elif (price < lowest_low_6h[i] and 
                  trend_12h_aligned[i] < 0):  # Bearish 12h trend
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals