#!/usr/bin/env python3
"""
Experiment #267: 6h Camarilla Pivot Reversal + Weekly Trend Filter

HYPOTHESIS: Camarilla pivot levels (R3/S3 for reversals, R4/S4 for breakouts) on 6h timeframe,
filtered by 1d supertrend and weekly trend alignment, capture high-probability mean reversion
and continuation moves. Weekly trend filter avoids counter-trend trades in strong markets,
while Camarilla levels provide precise entry/exit points. Targets 12-37 trades/year on 6h
to minimize fee drag while maintaining statistical significance. Works in bull/bear via
trend-aware filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_267_6h_camarilla_weekly_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Supertrend trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Supertrend on 1d data
    def calculate_supertrend(high_arr, low_arr, close_arr, period=10, multiplier=3.0):
        if len(high_arr) < period:
            return np.full_like(close_arr, np.nan), np.full_like(close_arr, np.nan)
        
        # True Range
        tr1 = high_arr[1:] - low_arr[1:]
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.concatenate([[high_arr[0] - low_arr[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # ATR
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        # Basic Upper and Lower Bands
        hl_avg = (high_arr + low_arr) / 2
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        # Final Bands
        final_upper = np.copy(upper_band)
        final_lower = np.copy(lower_band)
        for i in range(1, len(close_arr)):
            final_upper[i] = upper_band[i] if (upper_band[i] < final_upper[i-1] or close_arr[i-1] > final_upper[i-1]) else final_upper[i-1]
            final_lower[i] = lower_band[i] if (lower_band[i] > final_lower[i-1] or close_arr[i-1] < final_lower[i-1]) else final_lower[i-1]
        
        # Supertrend
        supertrend = np.full_like(close_arr, np.nan)
        for i in range(len(close_arr)):
            if i == 0:
                supertrend[i] = final_upper[i]
            elif supertrend[i-1] == final_upper[i-1]:
                supertrend[i] = final_upper[i] if close_arr[i] <= final_upper[i] else final_lower[i]
            else:
                supertrend[i] = final_lower[i] if close_arr[i] >= final_lower[i] else final_upper[i]
        
        # Trend direction: 1 = uptrend (price below supertrend), -1 = downtrend (price above supertrend)
        trend = np.where(close_arr > supertrend, -1, 1)
        return trend, supertrend
    
    trend_1d, _ = calculate_supertrend(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === HTF: 1w data for weekly trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(21) for trend
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    weekly_uptrend = df_1w['close'].values > ema_1w
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    
    # === 6h Indicators: Camarilla Pivot Levels (based on previous day) ===
    # Camarilla levels calculated from previous 1d OHLC
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h2 = np.full(n, np.nan)
    camarilla_l2 = np.full(n, np.nan)
    camarilla_h1 = np.full(n, np.nan)
    camarilla_l1 = np.full(n, np.nan)
    camarilla_close = np.full(n, np.nan)
    camarilla_pivot = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day's OHLC (approximate using available 6h data)
        # Since we don't have daily data in 6h timeframe, use rolling window of 4 periods (1 day = 4x6h)
        if i >= 4:
            prev_high = np.max(high[i-4:i])
            prev_low = np.min(low[i-4:i])
            prev_close = close[i-1]
            
            camarilla_pivot[i] = (prev_high + prev_low + prev_close) / 3
            range_val = prev_high - prev_low
            
            camarilla_h4[i] = camarilla_pivot[i] + (range_val * 1.1 / 2)
            camarilla_l4[i] = camarilla_pivot[i] - (range_val * 1.1 / 2)
            camarilla_h3[i] = camarilla_pivot[i] + (range_val * 1.1 / 4)
            camarilla_l3[i] = camarilla_pivot[i] - (range_val * 1.1 / 4)
            camarilla_h2[i] = camarilla_pivot[i] + (range_val * 1.1 / 6)
            camarilla_l2[i] = camarilla_pivot[i] - (range_val * 1.1 / 6)
            camarilla_h1[i] = camarilla_pivot[i] + (range_val * 1.1 / 12)
            camarilla_l1[i] = camarilla_pivot[i] - (range_val * 1.1 / 12)
            camarilla_close[i] = prev_close
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(24) for confirmation ===
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[24:] = volume[24:] / vol_ma_24[24:]
    vol_ratio[:24] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 100  # Ensure enough data for HTF trends, Camarilla, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(weekly_uptrend_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Trend Filter: Only trade in direction of weekly trend ---
        weekly_bullish = weekly_uptrend_aligned[i] > 0.5
        
        # --- 1d Supertrend Trend Filter ---
        trend_bullish = trend_1d_aligned[i] > 0  # 1 = uptrend
        trend_bearish = trend_1d_aligned[i] < 0  # -1 = downtrend
        
        # --- Volume Confirmation ---
        volume_confirm = vol_ratio[i] > 1.5
        
        # --- Camarilla Levels Conditions ---
        # Reversal at H3/L3 (fade extreme moves)
        reversal_long = (close[i] <= camarilla_l3[i]) and (low[i] < camarilla_l3[i]) and (close[i] > camarilla_l3[i])
        reversal_short = (close[i] >= camarilla_h3[i]) and (high[i] > camarilla_h3[i]) and (close[i] < camarilla_h3[i])
        
        # Breakout continuation at H4/L4 (strong moves)
        breakout_long = (close[i] > camarilla_h4[i]) and (volume_confirm)
        breakout_short = (close[i] < camarilla_l4[i]) and (volume_confirm)
        
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
                # Take profit at opposite H3/L3 level
                if camarilla_h3[i] > 0 and close[i] >= camarilla_h3[i]:
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
                # Take profit at opposite H3/L3 level
                if camarilla_l3[i] > 0 and close[i] <= camarilla_l3[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars (~18h) to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Long conditions:
        # 1. Reversal at L3 with weekly bullish bias OR
        # 2. Breakout above H4 with weekly bullish and 1d uptrend
        long_condition = (
            (reversal_long and weekly_bullish) or
            (breakout_long and weekly_bullish and trend_bullish)
        )
        
        # Short conditions:
        # 1. Reversal at H3 with weekly bearish bias OR
        # 2. Breakout below L4 with weekly bearish and 1d downtrend
        short_condition = (
            (reversal_short and not weekly_bullish) or
            (breakout_short and not weekly_bullish and trend_bearish)
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals