#!/usr/bin/env python3
"""
Experiment #227: 6h Donchian(20) breakout + 1d/1w weekly pivot + volume confirmation
HYPOTHESIS: Donchian breakouts on 6h aligned with weekly pivot direction (price > weekly pivot = bullish bias, price < weekly pivot = bearish bias) capture institutional flow. Volume confirmation (>2.0x average) filters weak breakouts. ATR stoploss (2.5x) manages risk. Discrete position sizing (0.25) balances return and fee drag. Target: 75-200 total trades over 4 years (19-50/year). Works in bull markets via breakout continuation with pivot bias and in bear markets via mean reversion at opposite band with pivot as dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_227_6h_donchian20_1d_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot points (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from prior week's daily OHLC
    # Weekly pivot = (PriorWeekHigh + PriorWeekLow + PriorWeekClose) / 3
    # R1 = 2*Pivot - PriorWeekLow, S1 = 2*Pivot - PriorWeekHigh
    # R2 = Pivot + (PriorWeekHigh - PriorWeekLow), S2 = Pivot - (PriorWeekHigh - PriorWeekLow)
    # R3 = PriorWeekHigh + 2*(Pivot - PriorWeekLow), S3 = PriorWeekLow - 2*(PriorWeekHigh - Pivot)
    
    # Shift by 1 week (5 trading days) to use prior week's data
    shift_period = 5  # 5 trading days = 1 week
    prior_week_high = df_1d['high'].shift(shift_period).values
    prior_week_low = df_1d['low'].shift(shift_period).values
    prior_week_close = df_1d['close'].shift(shift_period).values
    
    # Calculate weekly pivot levels
    weekly_pivot = (prior_week_high + prior_week_low + prior_week_close) / 3.0
    weekly_range = prior_week_high - prior_week_low
    r1 = 2 * weekly_pivot - prior_week_low
    s1 = 2 * weekly_pivot - prior_week_high
    r2 = weekly_pivot + weekly_range
    s2 = weekly_pivot - weekly_range
    r3 = prior_week_high + 2 * (weekly_pivot - prior_week_low)
    s3 = prior_week_low - 2 * (prior_week_high - weekly_pivot)
    
    # Align to 6h timeframe (shifted by 1 weekly bar for completed week only)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === HTF: 1w data for weekly trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    # Weekly EMA(21) for trend filter
    weekly_ema = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # === 6h Indicators: Donchian(20) channels ===
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 100  # Need enough for weekly calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(weekly_ema_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]
        breakout_down = low[i] < donch_lower[i-1]
        
        # --- Weekly Pivot Bias Conditions ---
        price_above_pivot = price > pivot_aligned[i]
        price_below_pivot = price < pivot_aligned[i]
        
        # --- Weekly Trend Filter ---
        weekly_trend_up = price > weekly_ema_aligned[i]
        weekly_trend_down = price < weekly_ema_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite band break with volume
                if breakout_down and volume_spike and price_below_pivot:
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
                # Exit on opposite band break with volume
                if breakout_up and volume_spike and price_above_pivot:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require volume spike + breakout conditions + pivot bias alignment
        if volume_spike:
            # Long: breakout up AND price above weekly pivot AND weekly trend up
            if breakout_up and price_above_pivot and weekly_trend_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout down AND price below weekly pivot AND weekly trend down
            elif breakout_down and price_below_pivot and weekly_trend_down:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals