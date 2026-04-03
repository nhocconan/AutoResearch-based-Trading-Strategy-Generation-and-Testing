#!/usr/bin/env python3
"""
Experiment #311: 6h Camarilla Pivot + 1d Volume Spike + ADX Trend Filter

HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
combined with 1d volume spike confirmation (>2.0x average) and ADX trend filter 
(ADX > 25) captures high-probability mean reversions in ranging markets and 
breakout continuations in trending markets. The 6h timeframe targets 12-37 trades/year 
(50-150 total over 4 years) to minimize fee drag while exploiting intraday extremes. 
Works in bull markets (breakouts at R4/S4 with volume) and bear markets (mean reversion 
at R3/S3 during rallies/pullbacks). Uses ATR-based stoploss for risk management.

IMPLEMENTATION NOTES:
- Uses discrete position sizing (0.25) to minimize churn
- Volume confirmation threshold set to 2.0x to balance signal quality and frequency
- ADX(14) > 25 ensures trending market conditions for breakout signals
- Mean reversion at R3/S3 only when ADX < 25 (ranging market)
- Warmup period set to 100 bars for stable indicator calculation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_311_6h_camarilla_1d_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume MA and ADX (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume MA(20) for spike detection
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.zeros(len(df_1d))
    vol_ratio_1d[20:] = df_1d['volume'].values[20:] / vol_ma_20_1d[20:]
    vol_ratio_1d[:20] = 1.0  # Neutral for warmup
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Calculate 1d ADX(14) for trend filter
    def calculate_adx(high_arr, low_arr, close_arr, period=14):
        if len(high_arr) < period + 1:
            return np.full_like(high_arr, np.nan)
        # True Range
        tr0 = high_arr - low_arr
        tr1 = np.abs(high_arr - np.concatenate([[close_arr[0]], close_arr[:-1]]))
        tr2 = np.abs(low_arr - np.concatenate([[close_arr[0]], close_arr[:-1]]))
        tr = np.maximum(tr0, np.maximum(tr1, tr2))
        # Directional Movement
        up_move = high_arr - np.concatenate([[high_arr[0]], high_arr[:-1]])
        down_move = np.concatenate([[low_arr[0]], low_arr[:-1]]) - low_arr
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        return adx
    
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators: Previous day's Camarilla levels ===
    # Camarilla levels based on previous 1d OHLC
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h2 = np.full(n, np.nan)
    camarilla_l2 = np.full(n, np.nan)
    camarilla_h1 = np.full(n, np.nan)
    camarilla_l1 = np.full(n, np.nan)
    camarilla_pivot = np.full(n, np.nan)
    
    # Calculate levels for each 6h bar using previous completed 1d bar
    for i in range(n):
        # Get index of previous completed 1d bar
        # Since we're using 6h bars, we need to map to 1d index
        # Simplified: use the 1d bar that ended before current 6h bar
        # We'll use the HTF data alignment approach but for OHLC
        pass  # Will calculate below using aligned arrays
    
    # Instead, calculate Camarilla levels from aligned 1d OHLC
    # Align 1d OHLC to 6s timeframe
    open_1d = get_htf_data(prices, '1d')['open'].values
    high_1d = get_htf_data(prices, '1d')['high'].values
    low_1d = get_htf_data(prices, '1d')['low'].values
    close_1d = get_htf_data(prices, '1d')['close'].values
    
    open_1d_aligned = align_htf_to_ltf(prices, df_1d, open_1d)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Shift by 1 to use previous completed day
    prev_close = np.concatenate([[np.nan], close_1d_aligned[:-1]])
    prev_high = np.concatenate([[np.nan], high_1d_aligned[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d_aligned[:-1]])
    
    # Camarilla formulas
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    camarilla_h4 = camarilla_pivot + (range_hl * 1.1 / 2)
    camarilla_l4 = camarilla_pivot - (range_hl * 1.1 / 2)
    camarilla_h3 = camarilla_pivot + (range_hl * 1.1 / 4)
    camarilla_l3 = camarilla_pivot - (range_hl * 1.1 / 4)
    camarilla_h2 = camarilla_pivot + (range_hl * 1.1 / 6)
    camarilla_l2 = camarilla_pivot - (range_hl * 1.1 / 6)
    camarilla_h1 = camarilla_pivot + (range_hl * 1.1 / 12)
    camarilla_l1 = camarilla_pivot - (range_hl * 1.1 / 12)
    
    # === 6h Indicators: ATR(14) for stoploss ===
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
    
    warmup = 100  # Increased warmup for stable HTF alignment and indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- ADX Trend Filter ---
        adx_value = adx_1d_aligned[i]
        is_trending = adx_value > 25
        is_ranging = adx_value <= 25
        
        # --- Price Position Relative to Camarilla Levels ---
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
            
            # Exit conditions based on market regime
            if is_trending:
                # In trending market: exit on Camarilla H4/L4 reversion (take profit)
                if position_side > 0 and price < camarilla_h4[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                elif position_side < 0 and price > camarilla_l4[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:
                # In ranging market: exit on Camarilla H3/L3 reversion (take profit)
                if position_side > 0 and price < camarilla_h3[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                elif position_side < 0 and price > camarilla_l3[i]:
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
        # Mean reversion in ranging market: fade at H3/L3
        long_mean_reversion = (is_ranging and 
                              price <= camarilla_l3[i] and 
                              volume_spike)
        
        short_mean_reversion = (is_ranging and 
                               price >= camarilla_h3[i] and 
                               volume_spike)
        
        # Breakout continuation in trending market: break at H4/L4
        long_breakout = (is_trending and 
                        price >= camarilla_h4[i] and 
                        volume_spike)
        
        short_breakout = (is_trending and 
                         price <= camarilla_l4[i] and 
                         volume_spike)
        
        if long_mean_reversion or long_breakout:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_mean_reversion or short_breakout:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals