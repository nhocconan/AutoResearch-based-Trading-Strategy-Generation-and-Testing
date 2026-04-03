#!/usr/bin/env python3
"""
Experiment #247: 6h Camarilla Pivot + Volume Spike + Regime Filter

HYPOTHESIS: Camarilla pivot levels from 1d timeframe provide high-probability reversal/continuation zones. 
Price approaching S3/R3 with volume spike (>2x average) and trending regime (ADX>25) signals continuation 
breakouts of S4/R4 levels. Price approaching S4/R4 with volume spike and ranging regime (ADX<20) signals 
mean reversion back toward the mean. This adaptive approach works in both bull (continuation breakouts) 
and bear (mean reversion in ranges) markets. 6h timeframe targets 12-37 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_247_6h_camarilla_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from prior 1d OHLC
    camarilla_s3 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_mean = np.full(n, np.nan)
    
    if len(df_1d) >= 2:  # Need at least 2 days of data
        # Align 1d data to LTF index for shifting
        df_1d_indexed = df_1d.set_index('open_time')
        
        # Calculate prior day's OHLC using shift(1) on the indexed series
        prior_day_high = df_1d_indexed['high'].shift(1).values
        prior_day_low = df_1d_indexed['low'].shift(1).values
        prior_day_close = df_1d_indexed['close'].shift(1).values
        
        # Calculate Camarilla levels for each prior day
        range_ = prior_day_high - prior_day_low
        camarilla_mean_daily = (prior_day_high + prior_day_low + prior_day_close) / 3.0
        camarilla_s3_daily = camarilla_mean_daily - (range_ * 1.1 / 4)
        camarilla_s4_daily = camarilla_mean_daily - (range_ * 1.1 / 2)
        camarilla_r3_daily = camarilla_mean_daily + (range_ * 1.1 / 4)
        camarilla_r4_daily = camarilla_mean_daily + (range_ * 1.1 / 2)
        
        # Create series aligned with 1d index
        camarilla_s3_series = pd.Series(index=df_1d_indexed.index, data=camarilla_s3_daily)
        camarilla_s4_series = pd.Series(index=df_1d_indexed.index, data=camarilla_s4_daily)
        camarilla_r3_series = pd.Series(index=df_1d_indexed.index, data=camarilla_r3_daily)
        camarilla_r4_series = pd.Series(index=df_1d_indexed.index, data=camarilla_r4_daily)
        camarilla_mean_series = pd.Series(index=df_1d_indexed.index, data=camarilla_mean_daily)
        
        # Align to LTF (6h) timeframe with shift(1) for completed bars only
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_series.values)
        camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_series.values)
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_series.values)
        camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_series.values)
        camarilla_mean_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mean_series.values)
    else:
        camarilla_s3_aligned = np.full(n, np.nan)
        camarilla_s4_aligned = np.full(n, np.nan)
        camarilla_r3_aligned = np.full(n, np.nan)
        camarilla_r4_aligned = np.full(n, np.nan)
        camarilla_mean_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: ADX(14) for regime detection ===
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed values
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di_14 = np.where(atr_14 > 0, (plus_dm_14 / atr_14) * 100, 0)
    minus_di_14 = np.where(atr_14 > 0, (minus_dm_14 / atr_14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di_14 + minus_di_14) > 0, 
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
        if (np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(adx_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Detection ---
        trending_regime = adx_14[i] > 25  # ADX > 25 = trending
        ranging_regime = adx_14[i] < 20   # ADX < 20 = ranging
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Price Proximity to Camarilla Levels (within 0.1% tolerance) ---
        tolerance = 0.001  # 0.1% tolerance
        near_s3 = abs(close[i] - camarilla_s3_aligned[i]) / close[i] <= tolerance
        near_s4 = abs(close[i] - camarilla_s4_aligned[i]) / close[i] <= tolerance
        near_r3 = abs(close[i] - camarilla_r3_aligned[i]) / close[i] <= tolerance
        near_r4 = abs(close[i] - camarilla_r4_aligned[i]) / close[i] <= tolerance
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss (using prior bar's ATR to avoid look-ahead)
            atr_stop = atr_14[i-1] if i > 0 else atr_14[i]
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_stop
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Camarilla mean reversion
                if close[i] > camarilla_mean_aligned[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_stop
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Camarilla mean reversion
                if close[i] < camarilla_mean_aligned[i]:
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
        # Long Continuation: Near R3 + volume spike + trending regime -> breakout R4
        long_continuation = near_r3 and volume_spike and trending_regime
        
        # Short Continuation: Near S3 + volume spike + trending regime -> breakdown S4
        short_continuation = near_s3 and volume_spike and trending_regime
        
        # Long Mean Reversion: Near S4 + volume spike + ranging regime -> revert to mean
        long_reversion = near_s4 and volume_spike and ranging_regime
        
        # Short Mean Reversion: Near R4 + volume spike + ranging regime -> revert to mean
        short_reversion = near_r4 and volume_spike and ranging_regime
        
        if long_continuation or long_reversion:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_continuation or short_reversion:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals