#!/usr/bin/env python3
"""
Experiment #047: 6h Williams %R(14) + 1d Elder Ray + Volume Spike
HYPOTHESIS: Williams %R identifies overbought/oversold extremes on 6h, while 1d Elder Ray (Bull/Bear Power) confirms the higher-timeframe trend direction. Volume spike (>2.0x average) filters false signals. This combination works in both bull and bear markets by taking counter-trend reversals aligned with the 1d trend's momentum exhaustion. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_047_6h_williamsr14_1d_elder_ray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Elder Ray (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Elder Ray on 1d: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    def calculate_ema(arr, period):
        return pd.Series(arr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    ema_13 = calculate_ema(df_1d['close'].values, 13)
    bull_power = df_1d['high'].values - ema_13
    bear_power = df_1d['low'].values - ema_13
    
    # Align Elder Ray to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # === 6h Indicators: Williams %R(14) ===
    def calculate_williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    williams_r = calculate_williams_r(high, low, close, 14)
    
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
    
    warmup = 50  # Warmup for indicator stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Williams %R Conditions ---
        wr_oversold = williams_r[i] < -80  # Oversold
        wr_overbought = williams_r[i] > -20  # Overbought
        
        # --- Elder Ray Trend: Determine 1d trend direction ---
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        # Bullish trend when Bull Power > 0 and rising (or Bear Power < 0)
        bullish_trend = bull_power_val > 0 and (i == warmup or bull_power_val > bull_power_aligned[i-1])
        # Bearish trend when Bear Power < 0 and falling (or Bull Power < 0)
        bearish_trend = bear_power_val < 0 and (i == warmup or bear_power_val < bear_power_aligned[i-1])
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
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
                # Exit on Williams %R reversal (overbought) with volume (profit taking)
                if wr_overbought and volume_spike:
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
                # Exit on Williams %R reversal (oversold) with volume (profit taking)
                if wr_oversold and volume_spike:
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
        # Long: Williams %R oversold AND 1d bullish trend AND volume spike
        if wr_oversold and bullish_trend and volume_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Williams %R overbought AND 1d bearish trend AND volume spike
        elif wr_overbought and bearish_trend and volume_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals