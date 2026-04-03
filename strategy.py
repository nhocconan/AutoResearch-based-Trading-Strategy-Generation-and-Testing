#!/usr/bin/env python3
"""
Experiment #075: 6h Williams %R(14) + 1d Elder Ray + Volume Spike
HYPOTHESIS: Williams %R identifies overbought/oversold extremes on 6h, while 1d Elder Ray (Bull/Bear Power) confirms the underlying trend direction. Volume spike (>2.0x average) filters false signals. This combination works in both bull and bear markets by fading extremes in the direction of the higher timeframe trend. ATR stoploss (2.5x) and minimum holding period (6 bars) reduce churn. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_075_6h_williamsr14_1d_elder_ray_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Elder Ray (Bull/Bear Power) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(13) on 1d close for Elder Ray
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    bull_power_1d = df_1d['high'].values - ema_13_1d
    bear_power_1d = df_1d['low'].values - ema_13_1d
    
    # Align to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # === 6h Indicators: Williams %R(14) ===
    def calculate_williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = np.zeros_like(close)
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr_14 = calculate_williams_r(high, low, close, 14)
    
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
        if (np.isnan(wr_14[i]) or np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Williams %R: Overbought/Oversold Conditions ---
        wr_oversold = wr_14[i] <= -80  # Oversold condition
        wr_overbought = wr_14[i] >= -20  # Overbought condition
        
        # --- Elder Ray Trend: Determine 1d trend direction ---
        bull_power = bull_power_1d_aligned[i]
        bear_power = bear_power_1d_aligned[i]
        elder_bull = bull_power > 0  # Bullish trend when Bull Power > 0
        elder_bear = bear_power < 0  # Bearish trend when Bear Power < 0
        
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
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 6 bars to reduce churn
            if bars_since_entry < 6:
                signals[i] = position_side * SIZE
                continue
            
            # Exit conditions: Williams %R returns to neutral territory
            if position_side > 0 and wr_14[i] >= -50:  # Long exit when WR rises above -50
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            elif position_side < 0 and wr_14[i] <= -50:  # Short exit when WR falls below -50
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Williams %R oversold AND Elder Ray bullish AND volume spike
        if wr_oversold and elder_bull and volume_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Williams %R overbought AND Elder Ray bearish AND volume spike
        elif wr_overbought and elder_bear and volume_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>