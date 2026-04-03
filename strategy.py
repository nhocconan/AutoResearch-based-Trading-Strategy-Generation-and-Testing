#!/usr/bin/env python3
"""
Experiment #147: 6h Camarilla Pivot + 1d Weekly Trend + Volume Confirmation

HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
filtered by 1d/1w trend direction (using EMA crossover) and volume confirmation 
capture high-probability reversals and continuations. The 1d EMA(50)/EMA(200) 
determines weekly bias: price above both = bullish bias (favor longs at S3/S4, 
breaks above R4), price below both = bearish bias (favor shorts at R3/R4, 
breaks below S4). Volume > 1.5x average confirms institutional participation. 
This structure works in bull markets (breakouts with volume) and bear markets 
(mean reversion at extremes + failed breaks). Targets 12-37 trades/year (50-150 
total over 4 years) to minimize fee drag while capturing significant moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_147_6h_camarilla_1d_weekly_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for 1d
    def camarilla_pivots(high_val, low_val, close_val):
        """Calculate Camarilla pivot levels"""
        pivot = (high_val + low_val + close_val) / 3
        range_val = high_val - low_val
        r4 = pivot + (range_val * 1.5 / 2)
        r3 = pivot + (range_val * 1.25 / 2)
        s3 = pivot - (range_val * 1.25 / 2)
        s4 = pivot - (range_val * 1.5 / 2)
        return r3, r4, s3, s4
    
    # Calculate pivots for each 1d bar
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_r4 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    camarilla_s4 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        r3, r4, s3, s4 = camarilla_pivots(
            df_1d['high'].iloc[i], 
            df_1d['low'].iloc[i], 
            df_1d['close'].iloc[i]
        )
        camarilla_r3[i] = r3
        camarilla_r4[i] = r4
        camarilla_s3[i] = s3
        camarilla_s4[i] = s4
    
    # Calculate 1d trend: EMA(50) > EMA(200) = bullish, EMA(50) < EMA(200) = bearish
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_200 = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    trend_bullish = ema_50 > ema_200
    trend_bearish = ema_50 < ema_200
    
    # Align HTF data to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish.astype(float))
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
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
    
    warmup = 200  # Ensure enough data for EMA(200), ATR, and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filters ---
        bullish_bias = trend_bullish_aligned[i] > 0.5
        bearish_bias = trend_bearish_aligned[i] > 0.5
        
        # --- Volume Confirmation: Require volume > 1.5x average ---
        volume_confirm = vol_ratio[i] > 1.5
        
        # --- Camarilla Levels ---
        r3 = camarilla_r3_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s3 = camarilla_s3_aligned[i]
        s4 = camarilla_s4_aligned[i]
        
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
                # Take profit at opposite Camarilla level
                if close[i] >= r3 and bullish_bias:  # Take profit at R3 in bullish bias
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
                # Take profit at opposite Camarilla level
                if close[i] <= s3 and bearish_bias:  # Take profit at S3 in bearish bias
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
        # Long conditions:
        # 1. Mean reversion: Price at S3 with bullish bias + volume
        # 2. Breakout: Price breaks above R4 with bullish bias + volume
        long_mean_reversion = (close[i] <= s3 and bullish_bias and volume_confirm)
        long_breakout = (close[i] > r4 and bullish_bias and volume_confirm)
        
        # Short conditions:
        # 1. Mean reversion: Price at R3 with bearish bias + volume
        # 2. Breakout: Price breaks below S4 with bearish bias + volume
        short_mean_reversion = (close[i] >= r3 and bearish_bias and volume_confirm)
        short_breakout = (close[i] < s4 and bearish_bias and volume_confirm)
        
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