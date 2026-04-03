#!/usr/bin/env python3
"""
Experiment #1731: 6h Camarilla Pivot + 1d Volume Spike + ADX Regime Filter
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) on 6h timeframe combined with 1d volume confirmation and ADX regime filter (ADX>25 for trending, ADX<20 for ranging) captures both mean reversion in ranges and breakout continuation in trends. This dual-regime approach should work in both bull and bear markets by adapting to market conditions. Target: 75-150 total trades over 4 years (19-37/year) with discrete position sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1731_6h_camarilla_pivot_vol_adx_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 1d data for volume and ADX (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.ones(len(close_1d))
    vol_ratio_1d[20:] = volume_1d[20:] / vol_ma_1d[20:]
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 1d Indicators: ADX(14) for regime filter ===
    def calculate_dmi(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        dx = np.zeros(len(high))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        return adx
    
    adx_1d = calculate_dmi(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators: Camarilla Pivot Levels ===
    # Calculate pivots from previous 6h bar (using rolling window)
    def calculate_camarilla(high, low, close):
        # Typical price for pivot
        pivot = (high + low + close) / 3.0
        range_val = high - low
        
        # Camarilla levels
        r3 = pivot + (range_val * 1.1 / 4.0)
        s3 = pivot - (range_val * 1.1 / 4.0)
        r4 = pivot + (range_val * 1.1 / 2.0)
        s4 = pivot - (range_val * 1.1 / 2.0)
        
        return pivot, r3, s3, r4, s4
    
    camarilla_data = [calculate_camarilla(high[i], low[i], close[i]) for i in range(n)]
    pivot = np.array([x[0] for x in camarilla_data])
    r3 = np.array([x[1] for x in camarilla_data])
    s3 = np.array([x[2] for x in camarilla_data])
    r4 = np.array([x[3] for x in camarilla_data])
    s4 = np.array([x[4] for x in camarilla_data])
    
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
    
    warmup = 20  # sufficient for volume MA and ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(pivot[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or
            np.isnan(r4[i]) or np.isnan(s4[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(atr[i])):
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
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        is_trending = adx_1d_aligned[i] > 25
        is_ranging = adx_1d_aligned[i] < 20
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        if volume_spike:
            if is_trending:
                # Trending regime: breakout continuation at R4/S4
                if price > r4[i]:  # Bullish breakout
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif price < s4[i]:  # Bearish breakdown
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # Ranging regime: mean reversion at R3/S3
                if price < s3[i] and price > pivot[i]:  # Oversold bounce from S3
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif price > r3[i] and price < pivot[i]:  # Overbought rejection from R3
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:
                # Transition regime (ADX 20-25): no trades
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals