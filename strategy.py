#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_1dTrend_Volume_Confirm_v3
# Hypothesis: Breakout of 1-day Camarilla R3/S3 levels on 4h chart with confirmation from 1-day EMA34 trend and volume spike. 
# Uses a dynamic position sizing based on volatility (ATR-based) to reduce risk in volatile markets and increase in stable conditions.
# Designed to work in both bull and bear markets by requiring trend alignment and volume confirmation. 
# Includes a minimum holding period of 12 bars to reduce trade frequency and avoid overtrading.
# Targets 20-30 trades/year to minimize fee drag.

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_Confirm_v3"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d Data (loaded ONCE) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Camarilla Pivot Levels (R3, S3) ===
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3 = pivot + (range_1d * 1.1 / 2)
    s3 = pivot - (range_1d * 1.1 / 2)
    
    # Align 1d levels to 4h
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 1d EMA34 Trend Filter ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === ATR for Volatility-Based Position Sizing ===
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # Require 1.5x average volume
    
    # === Signal Parameters ===
    base_position_size = 0.25  # Base 25% of capital per trade
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    holding_bars = 0
    
    # Start after warmup (covers EMA34 and ATR)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(ema34_1d_4h[i]) or np.isnan(volume_ok[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                holding_bars = 0
            else:
                signals[i] = 0.0
            continue
        
        # Dynamic position sizing: reduce size in high volatility, increase in low volatility
        # Normalize ATR relative to its 50-period median to avoid extreme values
        if i >= 50:
            atr_median = np.nanmedian(atr[i-50:i])
            if atr_median > 0:
                atr_ratio = atr[i] / atr_median
                # Invert ratio: low volatility -> higher size, high volatility -> lower size
                vol_factor = np.clip(1.0 / (atr_ratio + 0.5), 0.5, 2.0)
            else:
                vol_factor = 1.0
        else:
            vol_factor = 1.0
        
        position_size = base_position_size * vol_factor
        # Cap position size at 0.40 as per risk management rules
        position_size = min(position_size, 0.40)
        
        if position == 0:
            # Long: Break above R3 + above 1d EMA34 + volume spike
            if (close[i] > r3_4h[i] and 
                close[i] > ema34_1d_4h[i] and 
                volume_ok[i]):
                signals[i] = position_size
                position = 1
                holding_bars = 0
            # Short: Break below S3 + below 1d EMA34 + volume spike
            elif (close[i] < s3_4h[i] and 
                  close[i] < ema34_1d_4h[i] and 
                  volume_ok[i]):
                signals[i] = -position_size
                position = -1
                holding_bars = 0
        else:
            # Enforce minimum holding period (12 bars)
            holding_bars += 1
            if holding_bars < 12:
                signals[i] = position_size if position == 1 else -position_size
                continue
            
            # Exit: Price closes below/above opposite level
            if position == 1:
                if close[i] < s3_4h[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > r3_4h[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = -position_size
    
    return signals