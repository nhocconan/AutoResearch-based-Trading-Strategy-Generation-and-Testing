#!/usr/bin/env python3
"""
Experiment #4314: 1h timeframe with 4h Donchian(20) + 1d HMA(50) trend + volume confirmation
HYPOTHESIS: Using 1h as primary timeframe with 4h Donchian breakouts for entry timing and 1d HMA50 for trend filter reduces noise while maintaining sufficient trade frequency. Volume confirmation (>2.0x average) filters false breakouts. Session filter (08-20 UTC) avoids low-liquidity periods. Target: 60-150 total trades over 4 years (15-37/year) with position size 0.20. Works in bull via breakout continuation, in bear via shorting breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4314_1h_donchian20_4h_1d_hma_vol_v1"
timeframe = "1h"
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
    
    # === Precompute HTF: 4h Donchian Channel (20) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 20:
        donch_upper_4h = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
        donch_lower_4h = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
        donch_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_upper_4h)
        donch_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_lower_4h)
    else:
        donch_upper_4h_aligned = np.full(n, np.nan)
        donch_lower_4h_aligned = np.full(n, np.nan)
    
    # === Precompute HTF: 1d HMA(50) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        # Calculate HMA(50): WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half = 50 // 2
        sqrt_n = int(np.sqrt(50))
        wma_half = pd.Series(df_1d['close'].values).rolling(window=half, min_periods=half).mean().values
        wma_full = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_1d = pd.Series(raw_hma).rolling(window=sqrt_n, min_periods=sqrt_n).mean().values
        hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    else:
        hma_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 50)  # 4h Donchian, vol MA, ATR, 1d HMA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper_4h_aligned[i]) or np.isnan(donch_lower_4h_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or np.isnan(hma_1d_aligned[i])):
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
        # Require volume confirmation (> 2.0x average) to filter noise
        volume_confirm = vol_ratio[i] > 2.0
        
        if volume_confirm:
            # 4h Donchian breakout conditions (using previous bar's levels)
            breakout_up = close[i] > donch_upper_4h_aligned[i-1]  # Close above previous upper band
            breakout_dn = close[i] < donch_lower_4h_aligned[i-1]  # Close below previous lower band
            
            # 1d HMA50 trend filter
            price_above_hma = price > hma_1d_aligned[i]
            price_below_hma = price < hma_1d_aligned[i]
            
            # Long conditions: 4h Donchian breakout up + price above 1d HMA50
            long_entry = breakout_up and price_above_hma
            
            # Short conditions: 4h Donchian breakout down + price below 1d HMA50
            short_entry = breakout_dn and price_below_hma
            
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
        else:
            signals[i] = 0.0
    
    return signals