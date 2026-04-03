#!/usr/bin/env python3
"""
Experiment #011: 6h Camarilla Pivot + Volume Spike + 1d Trend Filter

HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) on 6h combined with 
1d trend filter (price above/below EMA50) and volume confirmation (>1.8x average) captures high-probability 
trades. In trending markets, we trade breakouts at R4/S4 with the trend. In ranging markets (price near EMA50), 
we fade extremes at R3/S3. Uses ATR-based stoploss (2.5x) and minimum 4-bar holding period. 
Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_011_6h_camarilla_pivot_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 6h Indicators: Previous day's Camarilla levels ===
    def calculate_camarilla(prev_high, prev_low, prev_close):
        """Calculate Camarilla pivot levels for given HLC"""
        range_val = prev_high - prev_low
        if range_val <= 0:
            return prev_close, prev_close, prev_close, prev_close, prev_close, prev_close, prev_close, prev_close
        
        pivot = (prev_high + prev_low + prev_close) / 3.0
        r4 = pivot + (range_val * 1.1 / 2.0)
        r3 = pivot + (range_val * 1.1 / 4.0)
        s3 = pivot - (range_val * 1.1 / 4.0)
        s4 = pivot - (range_val * 1.1 / 2.0)
        return pivot, r3, r4, s3, s4
    
    # Calculate Camarilla levels for each 6h bar using previous day's data
    camarilla_p = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    # Group by date to get previous day's OHLC
    dates = pd.to_datetime(prices['open_time']).date
    unique_dates = np.unique(dates)
    
    # Create mapping from date to previous day's OHLC
    prev_day_ohlc = {}
    for i, date in enumerate(unique_dates):
        if i > 0:
            prev_date = unique_dates[i-1]
            # Find indices for previous date
            prev_mask = (dates == prev_date)
            if np.any(prev_mask):
                prev_high = high[prev_mask].max()
                prev_low = low[prev_mask].min()
                prev_close = close[prev_mask][-1]  # Last close of previous day
                camarilla_p[dates == date], camarilla_r3[dates == date], camarilla_r4[dates == date], \
                camarilla_s3[dates == date], camarilla_s4[dates == date] = calculate_camarilla(prev_high, prev_low, prev_close)
    
    # For first day, use first bar's data as fallback
    if np.any(np.isnan(camarilla_p)):
        first_high = high[0]
        first_low = low[0]
        first_close = close[0]
        p, r3, r4, s3, s4 = calculate_camarilla(first_high, first_low, first_close)
        camarilla_p[np.isnan(camarilla_p)] = p
        camarilla_r3[np.isnan(camarilla_r3)] = r3
        camarilla_r4[np.isnan(camarilla_r4)] = r4
        camarilla_s3[np.isnan(camarilla_s3)] = s3
        camarilla_s4[np.isnan(camarilla_s4)] = s4
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
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
    
    warmup = 50  # Warmup for 1d EMA50 stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_r4[i]) or
            np.isnan(camarilla_s3[i]) or np.isnan(camarilla_s4[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- 1d Trend Filter: Only trade when price is clearly above/below EMA50 ---
        price = close[i]
        is_uptrend = price > ema50_1d_aligned[i] * 1.005  # 0.5% buffer above EMA50
        is_downtrend = price < ema50_1d_aligned[i] * 0.995  # 0.5% buffer below EMA50
        is_trending = is_uptrend or is_downtrend
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
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
                # Exit on R3/S3 mean reversion signal (contrarian exit in ranging)
                if not is_trending and ((position_side > 0 and high[i] >= camarilla_r3[i]) or 
                                       (position_side < 0 and low[i] <= camarilla_s3[i])):
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
                # Exit on R3/S3 mean reversion signal (contrarian exit in ranging)
                if not is_trending and ((position_side > 0 and high[i] >= camarilla_r3[i]) or 
                                       (position_side < 0 and low[i] <= camarilla_s3[i])):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 4 bars to reduce churn
            if bars_since_entry < 4:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if is_trending:
            # Trending market: Breakout at R4/S4 with volume spike
            if high[i] > camarilla_r4[i] and volume_spike and is_uptrend:
                # Long breakout above R4
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif low[i] < camarilla_s4[i] and volume_spike and is_downtrend:
                # Short breakdown below S4
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # Ranging market: Fade extremes at R3/S3
            if low[i] <= camarilla_s3[i] and volume_spike:
                # Long mean reversion from S3
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif high[i] >= camarilla_r3[i] and volume_spike:
                # Short mean reversion from R3
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
    
    return signals