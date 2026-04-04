#!/usr/bin/env python3
"""
Experiment #4324: 1d Donchian(20) breakout + 1w HMA(21) trend + volume confirmation
HYPOTHESIS: Daily Donchian breakouts capture swing momentum when aligned with weekly HMA21 trend (price > HMA21 for longs, < HMA21 for shorts) and confirmed by volume (>1.5x average). Weekly trend filter reduces whipsaw in bear markets while allowing trend-following in bull markets. ATR-based trailing stop (2.0x) for risk management. Position size 0.25 targets 30-100 total trades over 4 years (7-25/year). Works in bull via breakout continuation, in bear via shorting breakdowns and mean reversion during ranging periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4324_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
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
    
    # === Precompute HTF: 1w HMA21 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 21:
        # Calculate HMA(21): WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half = 21 // 2
        sqrt_n = int(np.sqrt(21))
        wma_half = pd.Series(df_1w['close'].values).rolling(window=half, min_periods=half).mean().values
        wma_full = pd.Series(df_1w['close'].values).rolling(window=21, min_periods=21).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_1w = pd.Series(raw_hma).rolling(window=sqrt_n, min_periods=sqrt_n).mean().values
        hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian Channel (20) ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # === 1d Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1d Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20, 20, 14, 21)  # Donchian, vol MA, ATR, 1w HMA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(hma_1w_aligned[i])):
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
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.5
        
        if volume_confirm:
            # Donchian breakout conditions (using previous bar's levels)
            breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
            breakout_dn = close[i] < donch_lower[i-1]  # Close below previous lower band
            
            # 1w HMA21 trend filter
            price_above_hma = price > hma_1w_aligned[i]
            price_below_hma = price < hma_1w_aligned[i]
            
            # Long conditions: Donchian breakout up + price above HMA21
            long_entry = breakout_up and price_above_hma
            
            # Short conditions: Donchian breakout down + price below HMA21
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