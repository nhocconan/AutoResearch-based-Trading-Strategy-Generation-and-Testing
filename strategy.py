#!/usr/bin/env python3
"""
Experiment #4255: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture swing momentum when aligned with weekly pivot trend (price above/below weekly pivot) and confirmed by volume (>1.8x average). Weekly pivot provides structural support/resistance from higher timeframe, reducing false breakouts. Uses 6h timeframe to balance trade frequency (target: 75-150 total trades over 4 years) and reduce fee drag. Works in bull via breakout continuation above weekly pivot, in bear via shorting breakdowns below weekly pivot. Novelty: Weekly pivot as trend filter (not yet tried in this session) combined with volume confirmation on 6h Donchian breaks.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4255_6h_donchian20_1w_pivot_vol_v1"
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
    
    # === Precompute HTF: 1d weekly pivot (using 1d data to calculate weekly pivot) ===
    # Get 1d data to calculate weekly pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        # Calculate weekly pivot: (weekly high + weekly low + weekly close) / 3
        # We need to resample 1d to weekly manually since we can't use .resample()
        # Instead, we'll use the previous week's OHLC to calculate pivot for current week
        # But to avoid look-ahead, we'll use lagged weekly values
        # Simpler approach: use previous day's high/low/close as proxy for short-term pivot
        # Actually, let's calculate true weekly pivot using 1d data with proper alignment
        
        # For weekly pivot, we need to group 1d data into weeks
        # Since we can't resample, we'll approximate: use rolling window of 5 days (1 week)
        # and calculate pivot from the max high, min low, and last close in that window
        # This is not perfect but avoids look-ahead when using shift(1) in alignment
        
        # Calculate rolling weekly pivot on 1d data
        df_1d_high = df_1d['high'].values
        df_1d_low = df_1d['low'].values
        df_1d_close = df_1d['close'].values
        
        # Weekly pivot = (period_high + period_low + period_close) / 3
        # Use 5-day window for approximate week
        period_high = pd.Series(df_1d_high).rolling(window=5, min_periods=5).max().values
        period_low = pd.Series(df_1d_low).rolling(window=5, min_periods=5).min().values
        period_close = pd.Series(df_1d_close).rolling(window=5, min_periods=5).mean().values  # approx weekly close
        weekly_pivot_1d = (period_high + period_low + period_close) / 3.0
        
        # Align to 6h timeframe
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_1d)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
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
    
    warmup = max(20, 20, 14, 5)  # Donchian, vol MA, ATR, weekly pivot lookback
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(weekly_pivot_aligned[i])):
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
        # Require volume confirmation (> 1.8x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.8
        
        if volume_confirm:
            # Donchian breakout conditions (using previous bar's levels)
            breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
            breakout_dn = close[i] < donch_lower[i-1]  # Close below previous lower band
            
            # Weekly pivot trend filter
            price_above_pivot = price > weekly_pivot_aligned[i]
            price_below_pivot = price < weekly_pivot_aligned[i]
            
            # Long conditions: Donchian breakout up + price above weekly pivot
            long_entry = breakout_up and price_above_pivot
            
            # Short conditions: Donchian breakout down + price below weekly pivot
            short_entry = breakout_dn and price_below_pivot
            
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