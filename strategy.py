#!/usr/bin/env python3
"""
Experiment #251: 6h Camarilla Pivot + Volume Spike + ADX Regime Filter

HYPOTHESIS: Camarilla pivot levels derived from 1d data provide intraday support/resistance. 
Entries occur on volume-confirmed breaks of R3/S3 (fade) or R4/S4 (continuation) only when 
ADX(14) > 25 indicates a trending regime. This combines mean reversion at extreme pivots 
with trend continuation at breakout levels, filtered by volume and regime to avoid false 
signals. 6h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize 
fee drag while capturing significant pivot-driven moves. Works in bull (continuation at R4/S4) 
and bear (fade at R3/S3, continuation at R4/S4 down) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_251_6h_camarilla_pivot_volume_adx_v1"
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
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    if len(df_1d) >= 2:  # Need at least 2 days of data
        # Align 1d data to LTF index for shifting
        df_1d_indexed = df_1d.set_index('open_time')
        
        # Get prior day's OHLC using shift(1)
        prior_day_high = df_1d_indexed['high'].shift(1).values
        prior_day_low = df_1d_indexed['low'].shift(1).values
        prior_day_close = df_1d_indexed['close'].shift(1).values
        
        # Calculate Camarilla levels for each prior day
        rng = prior_day_high - prior_day_low
        camarilla_r4_prior = prior_day_close + (rng * 1.1 / 2.0)
        camarilla_r3_prior = prior_day_close + (rng * 1.1 / 4.0)
        camarilla_s3_prior = prior_day_close - (rng * 1.1 / 4.0)
        camarilla_s4_prior = prior_day_close - (rng * 1.1 / 2.0)
        
        # Create series aligned with 1d index
        r4_series = pd.Series(index=df_1d_indexed.index, data=camarilla_r4_prior)
        r3_series = pd.Series(index=df_1d_indexed.index, data=camarilla_r3_prior)
        s3_series = pd.Series(index=df_1d_indexed.index, data=camarilla_s3_prior)
        s4_series = pd.Series(index=df_1d_indexed.index, data=camarilla_s4_prior)
        
        # Align to LTF (6h) timeframe with shift(1) for completed bars only
        camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, r4_series.values)
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, r3_series.values)
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, s3_series.values)
        camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, s4_series.values)
    else:
        camarilla_r4_aligned = np.full(n, np.nan)
        camarilla_r3_aligned = np.full(n, np.nan)
        camarilla_s3_aligned = np.full(n, np.nan)
        camarilla_s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: ADX(14) for regime filter ===
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    up_move = np.zeros(n)
    down_move = np.zeros(n)
    up_move[0] = 0
    down_move[0] = 0
    for i in range(1, n):
        up_move[i] = max(0, high[i] - high[i-1])
        down_move[i] = max(0, low[i-1] - low[i])
    
    # Smoothed ATR, +DM, -DM
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_plus = pd.Series(up_move).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_minus = pd.Series(down_move).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = np.where(atr_14 > 0, (dm_plus / atr_14) * 100, 0)
    di_minus = np.where(atr_14 > 0, (dm_minus / atr_14) * 100, 0)
    
    # ADX
    dx = np.where((di_plus + di_minus) > 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
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
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- ADX Regime Filter: Require trending market (ADX > 25) ---
        trending_regime = adx[i] > 25
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Price Levels ---
        price = close[i]
        r4 = camarilla_r4_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        s4 = camarilla_s4_aligned[i]
        
        # --- Camarilla-Based Entry Conditions ---
        # Long: 
        #   - Continuation breakout above R4 (bullish breakout)
        #   - OR fade from S3 (mean reversion up from strong support)
        long_breakout = price > r4 and trending_regime and volume_spike
        long_fade = price < s3 and trending_regime and volume_spike
        
        # Short:
        #   - Continuation breakdown below S4 (bearish breakdown)
        #   - OR fade from R3 (mean reversion down from strong resistance)
        short_breakout = price < s4 and trending_regime and volume_spike
        short_fade = price > r3 and trending_regime and volume_spike
        
        # --- Exit Logic (ATR-based stoploss) ---
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
                # Take profit at opposite Camarilla level
                if position_side > 0 and price < camarilla_s3_aligned[i]:
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
                # Take profit at opposite Camarilla level
                if position_side < 0 and price > camarilla_r3_aligned[i]:
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
        if long_breakout or long_fade:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_breakout or short_fade:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals