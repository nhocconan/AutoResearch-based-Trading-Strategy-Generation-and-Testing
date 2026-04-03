#!/usr/bin/env python3
"""
Experiment #235: 6h Camarilla Pivot + Weekly Trend + Volume Spike

HYPOTHESIS: 6h Camarilla pivot breakouts filtered by weekly trend direction (price > weekly EMA200 = bullish bias, 
price < weekly EMA200 = bearish bias) and volume spikes (>1.8x average) capture institutional interest at key 
mathematical levels. Weekly EMA200 provides structural trend filter to avoid counter-trend trades. 6h timeframe 
targets 12-37 trades/year (50-150 total over 4 years) with tight entry conditions to minimize fee drag. 
Works in bull markets (breakouts with volume in trend direction) and bear markets (failed reversions at pivot levels).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_235_6h_camarilla_weekly_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: Weekly data for EMA200 trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    weekly_ema200 = np.full(n, np.nan)
    
    if len(df_1w) >= 200:  # Need enough data for EMA200
        # Align weekly data to LTF index
        df_1w_indexed = df_1w.set_index('open_time')
        ema_200_series = pd.Series(df_1w_indexed['close'].values, index=df_1w_indexed.index).ewm(
            span=200, min_periods=200, adjust=False
        ).mean()
        weekly_ema200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_series.values)
    else:
        weekly_ema200_aligned = np.full(n, np.nan)
    
    # === HTF: Daily data for Camarilla pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from prior day's OHLC
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h2 = np.full(n, np.nan)
    camarilla_l2 = np.full(n, np.nan)
    camarilla_h1 = np.full(n, np.nan)
    camarilla_l1 = np.full(n, np.nan)
    camarilla_pivot = np.full(n, np.nan)
    
    if len(df_1d) >= 2:  # Need at least 2 days of data
        # Align 1d data to LTF index for shifting
        df_1d_indexed = df_1d.set_index('open_time')
        
        # Calculate prior day's OHLC using shift(1) on the indexed series
        prior_day_high = df_1d_indexed['high'].shift(1).values
        prior_day_low = df_1d_indexed['low'].shift(1).values
        prior_day_close = df_1d_indexed['close'].shift(1).values
        
        # Calculate Camarilla levels
        camarilla_pivot_daily = (prior_day_high + prior_day_low + prior_day_close) / 3.0
        daily_range = prior_day_high - prior_day_low
        
        camarilla_h4_daily = camarilla_pivot_daily + (daily_range * 1.1 / 2)
        camarilla_l4_daily = camarilla_pivot_daily - (daily_range * 1.1 / 2)
        camarilla_h3_daily = camarilla_pivot_daily + (daily_range * 1.1 / 4)
        camarilla_l3_daily = camarilla_pivot_daily - (daily_range * 1.1 / 4)
        camarilla_h2_daily = camarilla_pivot_daily + (daily_range * 1.1 / 6)
        camarilla_l2_daily = camarilla_pivot_daily - (daily_range * 1.1 / 6)
        camarilla_h1_daily = camarilla_pivot_daily + (daily_range * 1.1 / 12)
        camarilla_l1_daily = camarilla_pivot_daily - (daily_range * 1.1 / 12)
        
        # Create series aligned with 1d index
        camarilla_pivot_series = pd.Series(index=df_1d_indexed.index, data=camarilla_pivot_daily)
        camarilla_h4_series = pd.Series(index=df_1d_indexed.index, data=camarilla_h4_daily)
        camarilla_l4_series = pd.Series(index=df_1d_indexed.index, data=camarilla_l4_daily)
        camarilla_h3_series = pd.Series(index=df_1d_indexed.index, data=camarilla_h3_daily)
        camarilla_l3_series = pd.Series(index=df_1d_indexed.index, data=camarilla_l3_daily)
        camarilla_h2_series = pd.Series(index=df_1d_indexed.index, data=camarilla_h2_daily)
        camarilla_l2_series = pd.Series(index=df_1d_indexed.index, data=camarilla_l2_daily)
        camarilla_h1_series = pd.Series(index=df_1d_indexed.index, data=camarilla_h1_daily)
        camarilla_l1_series = pd.Series(index=df_1d_indexed.index, data=camarilla_l1_daily)
        
        # Align to LTF (6h) timeframe with shift(1) for completed bars only
        camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot_series.values)
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_series.values)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_series.values)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_series.values)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_series.values)
        camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2_series.values)
        camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2_series.values)
        camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1_series.values)
        camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1_series.values)
    else:
        camarilla_pivot_aligned = np.full(n, np.nan)
        camarilla_h4_aligned = np.full(n, np.nan)
        camarilla_l4_aligned = np.full(n, np.nan)
        camarilla_h3_aligned = np.full(n, np.nan)
        camarilla_l3_aligned = np.full(n, np.nan)
        camarilla_h2_aligned = np.full(n, np.nan)
        camarilla_l2_aligned = np.full(n, np.nan)
        camarilla_h1_aligned = np.full(n, np.nan)
        camarilla_l1_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
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
    
    warmup = 200  # Ensure enough data for weekly EMA200 and daily Camarilla
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(weekly_ema200_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Trend Filter: Price > weekly EMA200 = bullish bias, Price < weekly EMA200 = bearish bias ---
        price_above_weekly_ema = close[i] > weekly_ema200_aligned[i]
        price_below_weekly_ema = close[i] < weekly_ema200_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Camarilla Breakout Conditions ---
        breakout_h4 = close[i] > camarilla_h4_aligned[i]  # Break above H4
        breakdown_l4 = close[i] < camarilla_l4_aligned[i]  # Break below L4
        reversion_h3 = close[i] < camarilla_h3_aligned[i] and close[i] > camarilla_h2_aligned[i]  # Reversion from H3
        reversion_l3 = close[i] > camarilla_l3_aligned[i] and close[i] < camarilla_l2_aligned[i]  # Reversion from L3
        
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
                # Take profit at H3 level for longs
                if close[i] < camarilla_h3_aligned[i]:
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
                # Take profit at L3 level for shorts
                if close[i] > camarilla_l3_aligned[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Camarilla H4 breakout + volume spike + price above weekly EMA200
        long_condition = breakout_h4 and volume_spike and price_above_weekly_ema
        
        # Short: Camarilla L4 breakdown + volume spike + price below weekly EMA200
        short_condition = breakdown_l4 and volume_spike and price_below_weekly_ema
        
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