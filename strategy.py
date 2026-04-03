#!/usr/bin/env python3
"""
Experiment #1559: 6h Camarilla Pivot + 12h Volume Spike + ADX Trend Filter
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) on 6h timeframe combined with 12h volume confirmation (>2x average) and ADX(14)>25 for trending markets captures high-probability reversals in ranging markets and breakouts in trending markets. This dual-regime approach adapts to both bull and bear conditions by using volume as confirmation and ADX to filter false signals. Position size 0.25 balances return and drawdown. Target: 75-150 total trades over 4 years (19-38/year) by requiring confluence of pivot level, volume, and trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1559_6h_camarilla_pivot_vol_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume confirmation (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = np.ones(len(vol_12h))
    vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_12h[20:]
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # === HTF: 1d data for ADX trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    dm_plus_1d = np.zeros(len(close_1d))
    dm_minus_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        dm_plus_1d[i] = up_move if up_move > down_move and up_move > 0 else 0
        dm_minus_1d[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 6h Indicators: Camarilla Pivot Levels ===
    # Based on previous period's OHLC
    camarilla_s3 = np.zeros(n)
    camarilla_r3 = np.zeros(n)
    camarilla_s4 = np.zeros(n)
    camarilla_r4 = np.zeros(n)
    
    for i in range(1, n):
        # Previous 6h bar's OHLC
        prev_close = close[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        
        # Camarilla calculations
        range_val = prev_high - prev_low
        camarilla_s3[i] = prev_close - (range_val * 1.1 / 4)
        camarilla_r3[i] = prev_close + (range_val * 1.1 / 4)
        camarilla_s4[i] = prev_close - (range_val * 1.1 / 2)
        camarilla_r4[i] = prev_close + (range_val * 1.1 / 2)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # sufficient for Camarilla (needs 1 bar), volume MA, and ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_s3[i]) or np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s4[i]) or np.isnan(camarilla_r4[i]) or
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 2.0x average from 12h)
        volume_spike = vol_ratio_12h_aligned[i] > 2.0
        
        # ADX trend filter: require ADX > 25 for trending markets
        strong_trend = adx_aligned[i] > 25
        
        if volume_spike and strong_trend:
            # In strong trend, look for breakouts at R4/S4 levels
            if price > camarilla_r4[i]:  # Breakout above R4
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < camarilla_s4[i]:  # Breakdown below S4
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        elif volume_spike and not strong_trend:
            # In ranging market (ADX <= 25), look for mean reversion at R3/S3 levels
            if price < camarilla_r3[i] and price > camarilla_s3[i]:
                # In the mean reversion zone between R3 and S3
                # Look for rejection at extremes
                if price > camarilla_r3[i] * 0.999:  # Near R3, potential short
                    # Additional check: price rejected from R3 (close below open)
                    if close[i] < prices["open"].iloc[i]:
                        in_position = True
                        position_side = -1
                        entry_price = close[i]
                        bars_since_entry = 0
                        signals[i] = -SIZE
                elif price < camarilla_s3[i] * 1.001:  # Near S3, potential long
                    # Additional check: price rejected from S3 (close above open)
                    if close[i] > prices["open"].iloc[i]:
                        in_position = True
                        position_side = 1
                        entry_price = close[i]
                        bars_since_entry = 0
                        signals[i] = SIZE
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals