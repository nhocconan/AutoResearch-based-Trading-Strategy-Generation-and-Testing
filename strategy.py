#!/usr/bin/env python3
"""
Experiment #111: 6h Camarilla Pivot Fade/Breakout + Volume Spike + ADX Trend Filter

HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout) from 1d HTF provide 
institutional support/resistance. Fade at R3/S3 with volume confirmation captures mean reversion 
in ranging markets, while breakout at R4/S4 with volume and ADX>25 captures strong trends. 
This dual-regime approach works in both bull (breakouts) and bear (fades at resistance) markets. 
Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_111_6h_camarilla_pivot_fade_breakout_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot and ADX calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from prior day's OHLC
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    camarilla_R4 = np.full(n, np.nan)
    camarilla_S4 = np.full(n, np.nan)
    
    if len(df_1d) >= 2:  # Need at least 2 days of data
        # Align 1d data to LTF index for shifting
        df_1d_indexed = df_1d.set_index('open_time')
        
        # Calculate prior day's OHLC using shift(1)
        prior_day_high = df_1d_indexed['high'].shift(1).values
        prior_day_low = df_1d_indexed['low'].shift(1).values
        prior_day_close = df_1d_indexed['close'].shift(1).values
        
        # Calculate Camarilla levels for each prior day
        camarilla_R3_vals = prior_day_close + ((prior_day_high - prior_day_low) * 1.1 / 4)
        camarilla_S3_vals = prior_day_close - ((prior_day_high - prior_day_low) * 1.1 / 4)
        camarilla_R4_vals = prior_day_close + ((prior_day_high - prior_day_low) * 1.1 / 2)
        camarilla_S4_vals = prior_day_close - ((prior_day_high - prior_day_low) * 1.1 / 2)
        
        # Create series aligned with 1d index
        camarilla_R3_series = pd.Series(index=df_1d_indexed.index, data=camarilla_R3_vals)
        camarilla_S3_series = pd.Series(index=df_1d_indexed.index, data=camarilla_S3_vals)
        camarilla_R4_series = pd.Series(index=df_1d_indexed.index, data=camarilla_R4_vals)
        camarilla_S4_series = pd.Series(index=df_1d_indexed.index, data=camarilla_S4_vals)
        
        # Align to LTF (6h) timeframe with shift(1) for completed bars only
        camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3_series.values)
        camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3_series.values)
        camarilla_R4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R4_series.values)
        camarilla_S4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S4_series.values)
    else:
        camarilla_R3_aligned = np.full(n, np.nan)
        camarilla_S3_aligned = np.full(n, np.nan)
        camarilla_R4_aligned = np.full(n, np.nan)
        camarilla_S4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: ADX(14) for trend strength ===
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Calculate +DI and -DI
    plus_di_14 = np.where(atr_14 != 0, (plus_dm_14 / atr_14) * 100, 0)
    minus_di_14 = np.where(atr_14 != 0, (minus_dm_14 / atr_14) * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di_14 + minus_di_14) != 0, 
                  abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14) * 100, 0)
    adx_14 = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 50  # Ensure enough data for HTF Camarilla, ADX, and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(camarilla_R4_aligned[i]) or np.isnan(camarilla_S4_aligned[i]) or
            np.isnan(adx_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # === Camarilla Pivot Levels ===
        R3 = camarilla_R3_aligned[i]
        S3 = camarilla_S3_aligned[i]
        R4 = camarilla_R4_aligned[i]
        S4 = camarilla_S4_aligned[i]
        
        # === Volume Confirmation: Require volume spike (> 1.8x average) ===
        volume_spike = vol_ratio[i] > 1.8
        
        # === ADX Trend Filter: ADX > 20 indicates trending market ===
        trending = adx_14[i] > 20
        
        # === Exit Logic (ATR-based stoploss) ===
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss (2.0 * ATR)
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
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
        
        # === New Position Entry Logic (Only if Flat) ===
        # Fade at R3/S3 (mean reversion): Sell at R3, Buy at S3
        # Works in ranging markets (ADX <= 20) or at extremes
        fade_short = (close[i] >= R3) and volume_spike  # Sell at R3 resistance
        fade_long = (close[i] <= S3) and volume_spike   # Buy at S3 support
        
        # Breakout at R4/S4 (continuation): Buy at R4, Sell at S4
        # Works in trending markets (ADX > 20) with volume confirmation
        breakout_long = (close[i] >= R4) and volume_spike and trending
        breakout_short = (close[i] <= S4) and volume_spike and trending
        
        if fade_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif fade_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        elif breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals