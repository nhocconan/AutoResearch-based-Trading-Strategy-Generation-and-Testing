#!/usr/bin/env python3
"""
Experiment #091: 6h Camarilla Pivot Breakout + 1d Volume Spike + ATR Regime Filter

HYPOTHESIS: 6h Camarilla pivot breakouts (R4/S4 for continuation, R3/S3 for reversal) filtered by 1d volume spikes (>1.8x average) and ATR-based regime (ATR(6h)/ATR(1d) > 0.6 = volatile enough to trade) capture institutional breakout moves. Camarilla pivots derived from 1d OHLC provide mathematically derived support/resistance levels used by algo traders. Volume spike confirms participation, ATR regime filter avoids choppy low-volatility periods. 6h timeframe targets 12-37 trades/year (50-150 total) to minimize fee drag. Works in bull (R4 breakouts continuation) and bear (S4 breakdown continuation) markets. Uses ATR-based stoploss and time-based exit for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_091_6h_camarilla_pivot_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot and volume MA (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from 1d data (using prior day's OHLC)
    # Camarilla: R4 = Close + 1.5*(High-Low), R3 = Close + 1.1*(High-Low)
    #           S3 = Close - 1.1*(High-Low), S4 = Close - 1.5*(High-Low)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    camarilla_pivot = np.full(n, np.nan)  # (H+L+C)/3
    
    if len(df_1d) >= 2:  # Need at least 2 days of data
        # Align 1d data to LTF index for shifting
        df_1d_indexed = df_1d.set_index('open_time')
        
        # Calculate prior day's OHLC using shift(1) on the indexed series
        prior_day_high = df_1d_indexed['high'].shift(1).values
        prior_day_low = df_1d_indexed['low'].shift(1).values
        prior_day_close = df_1d_indexed['close'].shift(1).values
        
        # Calculate prior day's range
        prior_day_range = prior_day_high - prior_day_low
        
        # Calculate Camarilla levels for each prior day
        camarilla_r4_vals = prior_day_close + 1.5 * prior_day_range
        camarilla_r3_vals = prior_day_close + 1.1 * prior_day_range
        camarilla_s3_vals = prior_day_close - 1.1 * prior_day_range
        camarilla_s4_vals = prior_day_close - 1.5 * prior_day_range
        camarilla_pivot_vals = (prior_day_high + prior_day_low + prior_day_close) / 3.0
        
        # Create series aligned with 1d index
        camarilla_r4_series = pd.Series(index=df_1d_indexed.index, data=camarilla_r4_vals)
        camarilla_r3_series = pd.Series(index=df_1d_indexed.index, data=camarilla_r3_vals)
        camarilla_s3_series = pd.Series(index=df_1d_indexed.index, data=camarilla_s3_vals)
        camarilla_s4_series = pd.Series(index=df_1d_indexed.index, data=camarilla_s4_vals)
        camarilla_pivot_series = pd.Series(index=df_1d_indexed.index, data=camarilla_pivot_vals)
        
        # Align to LTF (6h) timeframe with shift(1) for completed bars only
        camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_series.values)
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_series.values)
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_series.values)
        camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_series.values)
        camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot_series.values)
    else:
        camarilla_r4_aligned = np.full(n, np.nan)
        camarilla_r3_aligned = np.full(n, np.nan)
        camarilla_s3_aligned = np.full(n, np.nan)
        camarilla_s4_aligned = np.full(n, np.nan)
        camarilla_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: ATR(14) for stoploss and regime filter ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_6h = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === HTF: 1d ATR for regime filter ===
    if len(df_1d) >= 2:
        df_1d_indexed = df_1d.set_index('open_time')
        tr_1d = np.zeros(len(df_1d))
        tr_1d[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
        for i in range(1, len(df_1d)):
            tr_1d[i] = max(
                df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
                abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
                abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
            )
        atr_1d_raw = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        atr_1d_series = pd.Series(index=df_1d_indexed.index, data=atr_1d_raw)
        atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_series.values)
    else:
        atr_1d_aligned = np.full(n, np.nan)
    
    # ATR regime: ATR(6h)/ATR(1d) > 0.6 = enough volatility to trade
    atr_ratio = np.zeros(n)
    valid_atr = (~np.isnan(atr_6h)) & (~np.isnan(atr_1d_aligned)) & (atr_1d_aligned > 0)
    atr_ratio[valid_atr] = atr_6h[valid_atr] / atr_1d_aligned[valid_atr]
    atr_ratio[~valid_atr] = 0.0
    volatile_regime = atr_ratio > 0.6
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    volume_spike = vol_ratio > 1.8  # Volume spike > 1.8x average
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position
    entry_bar_index = 0
    
    warmup = 50  # Ensure enough data for HTF calculations, ATR, etc.
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(atr_6h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime and Volume Filters ---
        in_volatile_regime = volatile_regime[i]
        vol_spike = volume_spike[i]
        
        # --- Camarilla Breakout Conditions ---
        breakout_r4 = close[i] > camarilla_r4_aligned[i]  # Bullish continuation
        breakdown_s4 = close[i] < camarilla_s4_aligned[i]  # Bearish continuation
        reversal_r3 = close[i] > camarilla_r3_aligned[i] and close[i-1] <= camarilla_r3_aligned[i-1]  # Cross above R3
        reversal_s3 = close[i] < camarilla_s3_aligned[i] and close[i-1] >= camarilla_s3_aligned[i-1]  # Cross below S3
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss (2 ATR)
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_6h[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Time-based exit: max 8 bars (48 hours) to avoid overstaying
                if bars_since_entry >= 8:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on reversal at R3/S3 (take profit at first support/resistance)
                if position_side > 0 and reversal_s3:  # Long exiting at S3
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_6h[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Time-based exit: max 8 bars (48 hours)
                if bars_since_entry >= 8:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on reversal at R3/S3 (take profit)
                if position_side < 0 and reversal_r3:  # Short exiting at R3
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
        # Require both regime filter and volume spike
        can_trade = in_volatile_regime and vol_spike
        
        if can_trade:
            # Long: R4 breakout (continuation) OR reversal from above R3
            long_condition = breakout_r4 or reversal_r3
            
            # Short: S4 breakdown (continuation) OR reversal from below S3
            short_condition = breakdown_s4 or reversal_s3
            
            if long_condition:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                entry_bar_index = i
                signals[i] = SIZE
            elif short_condition:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                entry_bar_index = i
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals