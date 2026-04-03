#!/usr/bin/env python3
"""
Experiment #262: 12h Camarilla Pivot + 1d Volume Spike + Chop Regime Filter

HYPOTHESIS: 12h Camarilla pivot levels (L3/L3/H3/H4) act as strong support/resistance 
zones where price often reverses or accelerates. Combined with 1d volume spikes (>2.0x 
average) to confirm institutional interest and a choppiness regime filter (CHOP > 61.8 
for ranging markets) to avoid false signals in strong trends. This strategy targets 
mean-reversion at pivot extremes in choppy markets and breakout continuations in 
trending markets, with discrete position sizing (0.25) to minimize fee drag. 
Designed for 12h timeframe to achieve 12-37 trades/year (50-150 total over 4 years) 
and works in both bull (breakouts with volume) and bear (mean reversion at pivots) 
markets. Uses ATR-based stoploss for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_262_12h_camarilla_1d_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume MA and chop regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.zeros_like(df_1d['close'])
    vol_ratio_1d[20:] = df_1d['volume'].values[20:] / vol_ma_20_1d[20:]
    vol_ratio_1d[:20] = 1.0  # Neutral for warmup
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 1d Indicators: Choppiness Index (CHOP) for regime detection ===
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        if len(close_arr) < period:
            return np.full_like(close_arr, 50.0)  # Neutral chop
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        sum_tr = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        highest_high = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        chop = 100 * np.log10(sum_tr / (atr * period)) / np.log10(period)
        chop = np.where((highest_high - lowest_low) > 0, chop, 50.0)
        return chop
    
    chop_1d = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 12h Indicators: Camarilla Pivot Levels (based on previous 1d bar) ===
    # Camarilla levels calculated from prior 1d daily OHLC
    # L4 = C - ((H-L)*1.1/2), L3 = C - ((H-L)*1.1/4), L2 = C - ((H-L)*1.1/6), L1 = C - ((H-L)*1.1/12)
    # H1 = C + ((H-L)*1.1/12), H2 = C + ((H-L)*1.1/6), H3 = C + ((H-L)*1.1/4), H4 = C + ((H-L)*1.1/2)
    # We use the close of the prior 1d bar to calculate levels for current 12h bar
    
    # Align 1d OHLC to 12h timeframe for pivot calculation
    df_1d_ohlc = get_htf_data(prices, '1d')[['open', 'high', 'low', 'close']]
    h_1d = df_1d_ohlc['high'].values
    l_1d = df_1d_ohlc['low'].values
    c_1d = df_1d_ohlc['close'].values
    
    # Calculate Camarilla levels from prior 1d bar
    diff = (h_1d - l_1d) * 1.1
    h4_1d = c_1d + diff / 2.0
    h3_1d = c_1d + diff / 4.0
    h2_1d = c_1d + diff / 6.0
    h1_1d = c_1d + diff / 12.0
    l1_1d = c_1d - diff / 12.0
    l2_1d = c_1d - diff / 6.0
    l3_1d = c_1d - diff / 4.0
    l4_1d = c_1d - diff / 2.0
    
    # Align 1d Camarilla levels to 12h timeframe (shifted by 1 for prior bar)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d_ohlc, h4_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d_ohlc, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d_ohlc, l3_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d_ohlc, l4_1d)
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 50  # Ensure enough data for HTF indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Chop > 61.8 = ranging (mean revert), Chop < 38.2 = trending (breakout) ---
        chop = chop_1d_aligned[i]
        is_ranging = chop > 61.8
        is_trending = chop < 38.2
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Price Levels ---
        price = close[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Exit conditions based on regime
            if is_ranging:
                # In ranging market: mean revert at L3/H3
                if position_side > 0 and price < h3_1d_aligned[i]:  # Long exit at H3
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                elif position_side < 0 and price > l3_1d_aligned[i]:  # Short exit at L3
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:
                # In trending market: trail with ATR or exit on opposite pivot touch
                if position_side > 0 and price < l3_1d_aligned[i]:  # Long exit if breaks L3
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                elif position_side < 0 and price > h3_1d_aligned[i]:  # Short exit if breaks H3
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Ranging market: Mean reversion at extreme pivot levels
        if is_ranging and volume_spike:
            # Long at L4 with rejection (price > L4 and closing back above L3)
            if price <= l4_1d_aligned[i] and close[i] > l3_1d_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short at H4 with rejection (price >= H4 and closing back below H3)
            elif price >= h4_1d_aligned[i] and close[i] < h3_1d_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
        
        # Trending market: Breakout continuation at pivot levels
        elif is_trending and volume_spike:
            # Long breakout above H3 with volume
            if price > h3_1d_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short breakdown below L3 with volume
            elif price < l3_1d_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals