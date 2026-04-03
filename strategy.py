#!/usr/bin/env python3
"""
Experiment #071: 6h Williams %R(14) + 1d Elder Ray + Volume Confirmation
HYPOTHESIS: Williams %R identifies overbought/oversold conditions on 6h, while 1d Elder Ray (Bull/Bear Power) 
provides trend filter. Volume confirmation (>1.5x average) ensures participation. 
Target: 75-150 trades over 4 years. Works in both bull/bear via trend-aligned mean reversion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_071_6h_williamsr14_1d_elder_ray_vol_v1"
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
    
    # Calculate Elder Ray on 1d: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(df_1d['close'].values).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # === 6h Indicators: Williams %R(14) ===
    def calculate_williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr_14 = calculate_williams_r(high, low, close, 14)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
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
    
    warmup = 30  # Warmup for Williams %R and EMA stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(wr_14[i]) or np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss (2.0x ATR)
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 3.0x ATR
                if high[i] > entry_price + 3.0 * atr_14[i]:
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
                # Take profit at 3.0x ATR
                if low[i] < entry_price - 3.0 * atr_14[i]:
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
        # Williams %R levels: > -20 = overbought, < -80 = oversold
        wr_overbought = wr_14[i] > -20
        wr_oversold = wr_14[i] < -80
        
        # Volume confirmation: require > 1.5x average volume
        volume_confirm = vol_ratio[i] > 1.5
        
        # Elder Ray trend filter: Bull Power > 0 = uptrend, Bear Power < 0 = downtrend
        bull_power = bull_power_1d_aligned[i]
        bear_power = bear_power_1d_aligned[i]
        uptrend = bull_power > 0
        downtrend = bear_power < 0
        
        # Long: Oversold + Uptrend + Volume
        if wr_oversold and uptrend and volume_confirm:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Overbought + Downtrend + Volume
        elif wr_overbought and downtrend and volume_confirm:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals