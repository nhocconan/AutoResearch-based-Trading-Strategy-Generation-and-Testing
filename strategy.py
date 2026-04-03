#!/usr/bin/env python3
"""
Experiment #217: 4h Donchian Breakout + HMA Trend + Volume Confirmation

HYPOTHESIS: Donchian(20) breakouts in the direction of 4h HMA(21) trend with volume confirmation
capture strong momentum moves. In bull markets, breakouts above upper channel with uptrend
(HMA rising) go long. In bear markets, breakouts below lower channel with downtrend
(HMA falling) go short. Volume spike (>1.5x MA20) confirms conviction. Uses 4h timeframe
to target 19-50 trades/year (75-200 total over 4 years) minimizing fee drag. ATR(14) stoploss
at 2.5x manages risk. Works in both bull (strong breakouts) and bear (failed reversals
at channel edges) markets by following the trend defined by HMA slope.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_217_4h_donchian_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX for regime detection (trending vs ranging)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        return adx
    
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 4h Indicators: Donchian Channel (20) ===
    donchian_window = 20
    dc_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    dc_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # === 4h Indicators: HMA(21) for trend direction ===
    def hull_moving_average(arr, period):
        half = arr.shape[0] // 2
        sqrt = int(np.sqrt(period))
        if half < 1 or sqrt < 1:
            return np.full_like(arr, np.nan)
        wma2 = pd.Series(arr).ewm(span=half, min_periods=half, adjust=False).mean()
        wma1 = pd.Series(arr).ewm(span=period, min_periods=period, adjust=False).mean()
        raw = 2 * wma2 - wma1
        hma = pd.Series(raw).ewm(span=sqrt, min_periods=sqrt, adjust=False).mean()
        return hma.values
    
    hma_21 = hull_moving_average(close, 21)
    # HMA slope: rising if current > previous, falling if current < previous
    hma_rising = np.zeros(n, dtype=bool)
    hma_falling = np.zeros(n, dtype=bool)
    hma_rising[1:] = hma_21[1:] > hma_21[:-1]
    hma_falling[1:] = hma_21[1:] < hma_21[:-1]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 100  # Warmup for Donchian, HMA, ADX stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or 
            np.isnan(hma_21[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- 1d Regime Filter: ADX > 25 = trending (we only trade in trending markets) ---
        is_trending = adx_1d_aligned[i] > 25
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Price Levels ---
        price = close[i]
        upper = dc_high[i]
        lower = dc_low[i]
        
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
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Exit on Donchian opposite touch (mean reversion within channel)
            if position_side > 0 and price < lower:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            if position_side < 0 and price > upper:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long breakout: Price > Upper Donchian + volume spike + trending + HMA rising
        long_breakout = (price > upper) and volume_spike and is_trending and hma_rising[i]
        
        # Short breakout: Price < Lower Donchian + volume spike + trending + HMA falling
        short_breakout = (price < lower) and volume_spike and is_trending and hma_falling[i]
        
        if long_breakout:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_breakout:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals