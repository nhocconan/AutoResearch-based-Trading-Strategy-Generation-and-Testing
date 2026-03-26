#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian Breakout + Volume + ATR Momentum + 1d Trend

HYPOTHESIS: Price BREAKING OUT of 20-period Donchian channel (not just 
touching) with volume confirmation and ATR momentum filter produces high-
quality entries that work in both bull and bear markets.

WHY THIS WORKS:
- Breakout = price penetrating established range = institutional move
- Volume confirms institutional participation, not retail noise  
- ATR momentum filter eliminates choppy, low-vol breakouts (false signals)
- 1d HMA ensures we're aligned with the larger trend direction

KEY FIX from failures:
- #016 (2443 trades!) used "within 2 ATR of pivot" = way too loose
- This strategy uses actual BREAKOUT (close > 20-bar high) = tight entries
- All 4 conditions must align = <100 trades/year

TARGET: 75-150 total trades over 4 years (~19-37/year)
DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe=1.382, 95tr)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_atr_1d_trend_v3"
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
    
    # === LOAD 1d DATA ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # === CALCULATE 4h INDICATORS ===
    period = 20  # Donchian period
    
    # Donchian channel
    donch_high = pd.Series(high).rolling(window=period, min_periods=period).max().shift(1).values
    donch_low = pd.Series(low).rolling(window=period, min_periods=period).min().shift(1).values
    
    # ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    # ATR momentum ratio (filter out choppy breakouts)
    atr_ratio = atr_14 / np.where(atr_30 > 0, atr_30, 1)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    warmup = max(50, period + 30)
    
    for i in range(warmup, n):
        # Skip if any indicator not ready
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_ratio[i]) or atr_ratio[i] <= 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ma[i]) or vol_ma[i] <= 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === CONDITION 1: Price BREAKOUT (close > prior 20-bar high) ===
        long_breakout = close[i] > donch_high[i]
        
        # === CONDITION 2: Price BREAKDOWN (close < prior 20-bar low) ===
        short_breakout = close[i] < donch_low[i]
        
        # === CONDITION 3: Volume confirmation (1.3x 20-bar MA) ===
        vol_confirm = volume[i] > vol_ma[i] * 1.3
        
        # === CONDITION 4: ATR momentum filter (>1.05 = not choppy) ===
        atr_momentum = atr_ratio[i] > 1.05
        
        # === CONDITION 5: 1d trend bias ===
        trend_bull = close[i] > hma_1d_aligned[i]
        trend_bear = close[i] < hma_1d_aligned[i]
        
        # === ENTRY LOGIC ===
        # ALL conditions must align for a valid signal
        
        if long_breakout and trend_bull and vol_confirm and atr_momentum:
            # Long entry on breakout + bull trend + volume + momentum
            signals[i] = SIZE
        
        elif short_breakout and trend_bear and vol_confirm and atr_momentum:
            # Short entry on breakdown + bear trend + volume + momentum
            signals[i] = -SIZE
        
        # If no breakout signal, flat (no partial positions)
        # This is a breakout-only strategy - no ranging entries
    
    return signals