#!/usr/bin/env python3
"""
Experiment #1339: 6h Camarilla Pivot Breakout + 12h Trend + Volume Confirmation
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) on 6h timeframe capture institutional order flow. 
Trend filter from 12h timeframe ensures alignment with intermediate-term momentum. Volume confirmation (>1.5x average) filters for participation. 
In ranging markets (ADX < 25), fade at R3/S3. In trending markets (ADX >= 25), breakout at R4/S4 continues the trend. 
Designed to work in both bull (breakouts continue) and bear (breakdowns continue) markets by adapting to regime. 
Uses ATR-based stoploss for risk management. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1339_6h_camarilla_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # EMA(20) trend: price > EMA20 = uptrend, < = downtrend
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    trend_12h = np.zeros(len(close_12h))
    trend_12h[20:] = np.where(close_12h[20:] > ema_12h[20:], 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Calculate Camarilla pivot levels for each 1d bar
    camarilla_r4 = np.zeros(len(close_1d))
    camarilla_r3 = np.zeros(len(close_1d))
    camarilla_s3 = np.zeros(len(close_1d))
    camarilla_s4 = np.zeros(len(close_1d))
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    camarilla_r4 = pivot + (range_1d * 1.1 / 2)
    camarilla_r3 = pivot + (range_1d * 1.1 / 4)
    camarilla_s3 = pivot - (range_1d * 1.1 / 4)
    camarilla_s4 = pivot - (range_1d * 1.1 / 2)
    # Align to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === 6h Indicators: ADX(14) for regime detection ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # +DM and -DM
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed +DM, -DM, TR
    atr_period = 14
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # +DI and -DI
    plus_di = np.where(tr_smooth > 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth > 0, 100 * minus_dm_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # sufficient for ADX and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_12h_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(adx[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Regime filter: ADX < 25 = ranging (mean reversion), ADX >= 25 = trending (breakout)
            if adx[i] < 25:  # Ranging market: fade at R3/S3
                if price >= camarilla_r3_aligned[i] and trend_12h_aligned[i] < 0:  # Short at R3 in downtrend
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                elif price <= camarilla_s3_aligned[i] and trend_12h_aligned[i] > 0:  # Long at S3 in uptrend
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                else:
                    signals[i] = 0.0
            else:  # Trending market: breakout at R4/S4
                if price > camarilla_r4_aligned[i] and trend_12h_aligned[i] > 0:  # Breakout above R4 in uptrend
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif price < camarilla_s4_aligned[i] and trend_12h_aligned[i] < 0:  # Breakdown below S4 in downtrend
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