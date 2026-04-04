#!/usr/bin/env python3
"""
Experiment #5549: 4h Donchian(20) breakout + 1d volume spike + chop regime filter
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts with volume > 2.0x average and 
Choppiness Index < 38.2 (trending regime) capture sustainable trend moves. 
The chop filter avoids false breakouts in ranging markets, while volume confirmation 
ensures institutional participation. Discrete sizing (0.25) minimizes fee drag. 
Target: 19-50 trades/year (75-200 total over 4 years) with ATR-based trailing stop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5549_4h_donchian20_1d_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for volume average (used for spike detection) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 20:
        # Calculate 20-day average volume from daily data
        vol_1d = df_1d['volume'].values
        avg_vol_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        # Align to LTF (4h) with shift(1) for completed bars only
        avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    else:
        # Fallback to LTF volume average if insufficient 1d data
        avg_vol_1d_aligned = np.full(n, 1.0)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation (vs 1d average) ===
    volume_ratio = volume / np.where(avg_vol_1d_aligned > 0, avg_vol_1d_aligned, 1)
    
    # === 4h Indicators: Choppiness Index (14-period) for regime filter ===
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        """Calculate Choppiness Index: higher = ranging, lower = trending"""
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        atr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        hh = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        
        # Avoid division by zero
        range_hl = hh - ll
        chop = np.where(
            (range_hl > 0) & (atr_sum > 0),
            100 * np.log10(atr_sum / range_hl) / np.log10(window),
            50.0  # neutral when undefined
        )
        return chop
    
    chop = calculate_chop(high, low, close, window=14)
    
    # === 4h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 14)  # Donchian, volume avg, chop, ATR warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (21:00-23:00 UTC) ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit: stoploss OR Donchian lower band break OR chop > 61.8 (range regime)
                if price <= stop_price or price <= donchian_low[i] or chop[i] > 61.8:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit: stoploss OR Donchian upper band break OR chop > 61.8 (range regime)
                if price >= stop_price or price >= donchian_high[i] or chop[i] > 61.8:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 2.0
        trending_regime = chop[i] < 38.2  # Chop < 38.2 = strong trend
        
        # Long: bullish breakout with volume confirmation in trending regime
        long_entry = breakout_up and volume_confirmed and trending_regime
        # Short: bearish breakout with volume confirmation in trending regime
        short_entry = breakout_down and volume_confirmed and trending_regime
        
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