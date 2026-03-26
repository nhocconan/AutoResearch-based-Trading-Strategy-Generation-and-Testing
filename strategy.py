#!/usr/bin/env python3
"""
Experiment #006: 4h Donchian Breakout + Volume Spike + ATR Regime

HYPOTHESIS: Donchian(20) breakout captures structural market moves when 
institutions break key levels. Volume confirms institutional involvement.
ATR regime ensures we only trade when volatility is expanding (momentum building).

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Breakouts occur in both directions - we trade both
- In bull: long breakouts above upper Donchian
- In bear: short breakouts below lower Donchian  
- ATR regime filter avoids whipsaws during low-vol consolidation

WHY THIS SHOULD WORK (vs previous failures):
- DB has MULTIPLE Donchian winners (4h: 1.10-1.38, 6h: 1.46)
- Previous attempts failed with 0 trades or negative Sharpe
- Key difference: ENTRY AT EXACT BREAKOUT (bar closes outside channel)
- Previous "near" entries caused overlap and extra trades

KEY DESIGN:
1. Donchian(20) - price must CLOSE outside channel (strict)
2. Volume 1.8x 20-avg (strict)
3. ATR(14) expanding (current > 1.2 * 10-bar ATR MA)
4. 1d HMA(21) for trend bias only (filter counter-trend entries)
5. Stop at opposite Donchian band (tight, structural)
6. Size: 0.30

TARGET: 75-150 total trades over 4 years.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_atr_regime_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data for trend bias
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channel (20 periods = ~5 days on 4h)
    donchian_period = 20
    upper_donchian = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_donchian = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ATR regime: current ATR > 1.2 * 10-bar ATR MA (volatility expanding)
    atr_ma = pd.Series(atr_14).rolling(window=10, min_periods=10).mean().values
    atr_regime = atr_14 / np.where(atr_ma > 0, atr_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup
    warmup = max(donchian_period + 20, 100)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === ENTRY CONDITIONS ===
        
        # 1. Donchian breakout: price CLOSES outside channel
        bullish_breakout = close[i] > upper_donchian[i]
        bearish_breakout = close[i] < lower_donchian[i]
        
        # 2. Volume confirmation (strict: 1.8x)
        vol_confirm = vol_ratio[i] > 1.8
        
        # 3. ATR regime (volatility expanding)
        atr_expanding = atr_regime[i] > 1.2
        
        # 4. 1d HMA trend bias (filter counter-trend entries)
        # Only use as filter, not entry trigger
        price_above_1d_hma = (not np.isnan(hma_1d_aligned[i]) and 
                              close[i] > hma_1d_aligned[i])
        price_below_1d_hma = (not np.isnan(hma_1d_aligned[i]) and 
                              close[i] < hma_1d_aligned[i])
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: bullish breakout + volume + ATR expanding + trend bias
            if bullish_breakout and vol_confirm and atr_expanding:
                # Allow longs regardless of trend (trend just biases)
                # But filter if in strong downtrend (price far below 1d HMA)
                if price_above_1d_hma or (not np.isnan(hma_1d_aligned[i]) and 
                                         close[i] > 0.8 * hma_1d_aligned[i]):
                    desired_signal = SIZE
            
            # SHORT: bearish breakout + volume + ATR expanding + trend bias
            if bearish_breakout and vol_confirm and atr_expanding:
                # Filter if in strong uptrend
                if price_below_1d_hma or (not np.isnan(hma_1d_aligned[i]) and 
                                          close[i] < 1.2 * hma_1d_aligned[i]):
                    desired_signal = -SIZE
        
        # === TRAILING STOP (structural: opposite Donchian band) ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                # Trail stop to lower Donchian
                trail_stop = lower_donchian[i]
                if low[i] < trail_stop:
                    desired_signal = 0.0
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Trail stop to upper Donchian
                trail_stop = upper_donchian[i]
                if high[i] > trail_stop:
                    desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                # Set stop at opposite Donchian band
                if position_side > 0:
                    stop_price = lower_donchian[i]
                else:
                    stop_price = upper_donchian[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals