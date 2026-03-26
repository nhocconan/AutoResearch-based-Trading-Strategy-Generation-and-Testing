#!/usr/bin/env python3
"""
Experiment #1620: 4h Donchian Breakout + Volume + 1d HMA Trend

HYPOTHESIS: Keep it SIMPLE. Donchian breakout is the proven core pattern 
(Sharpe 1.10-1.38 on SOLUSDT). Adding more indicators doesn't help — it adds
conflicts that either overtrade or miss trades entirely.

Why this should work in BOTH bull AND bear:
- Bull: Price breaks Donchian upper → momentum continues → captures rallies
- Bear: Short when price breaks Donchian lower → captures crash momentum
- 1d HMA adds trend filter (bull market = only long, bear = only short)
- Volume confirms breakout legitimacy (reduces false breakouts)

Key design decisions:
1. SIMPLE: Only 3 conditions per entry (Donchian break + 1d trend + volume)
2. NO Fisher/CRSI/Alligator — these add complexity without improving Sharpe
3. Size: 0.30 (discrete)
4. ATR stoploss: 2.5x (simple, proven)
5. Target: 75-200 trades/4yr (proven range from successful strategies)

This is essentially: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe 1.38)
minus the RSI filter (which often conflicts with Donchian signal).

Target: Sharpe > 0.6, trades 75-150 train, trades >= 10 test
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_volume_simple_1d_v1"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - price channel breakout system"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_volume_confirm(volume, period=20):
    """Volume confirmation: current vol > 1.5x 20-period average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = np.full(n, np.nan, dtype=np.float64)
    mask = vol_ma > 0
    ratio[mask] = volume[mask] / vol_ma[mask]
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=48)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    vol_ratio = calculate_volume_confirm(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete size
    
    # Warmup period
    min_bars = 50
    
    # Previous signal for tracking changes
    prev_signal = 0.0
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # === 1d TREND DIRECTION ===
        # Use closed 1d bars only (aligned array handles shift)
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT ===
        # Breakout = close above previous upper band (confirmed close, not intrabar)
        donch_break_long = close[i] > donch_upper[i-1] if i > 0 and not np.isnan(donch_upper[i-1]) else False
        donch_break_short = close[i] < donch_lower[i-1] if i > 0 and not np.isnan(donch_lower[i-1]) else False
        
        # === VOLUME CONFIRMATION ===
        # Volume > 1.5x 20-period average = breakout confirmed
        vol_confirm = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        # Keep it simple: Trend + Breakout + Volume = Entry
        desired_signal = 0.0
        
        # LONG: Price above 1d HMA (bull trend) + Donchian breakout + Volume confirm
        if price_above_1d and donch_break_long and vol_confirm:
            desired_signal = SIZE
        
        # SHORT: Price below 1d HMA (bear trend) + Donchian breakdown + Volume confirm
        elif price_below_1d and donch_break_short and vol_confirm:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK ===
        # 2.5x ATR stoploss
        if prev_signal > 0 and desired_signal == 0:
            # Stopped out or exited
            pass
        elif prev_signal < 0 and desired_signal == 0:
            # Stopped out or exited
            pass
        
        # Check ATR stoploss while in position
        if prev_signal > 0:  # Long position
            stop_price = close[i] - 2.5 * atr_14[i]
            if low[i] < stop_price:
                desired_signal = 0.0  # Stoploss hit
        
        if prev_signal < 0:  # Short position
            stop_price = close[i] + 2.5 * atr_14[i]
            if high[i] > stop_price:
                desired_signal = 0.0  # Stoploss hit
        
        # === DISCRETIZE ===
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        signals[i] = final_signal
        prev_signal = final_signal
    
    return signals