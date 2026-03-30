#!/usr/bin/env python3
"""
Experiment #028: 1d Golden/Death Cross + SMA(200) + Volume + ATR Squeeze

HYPOTHESIS: Golden/Death Cross is the most proven long-term trend signal in markets.
BTC's 2022 crash (77%) was preceded by Death Cross on monthly. This strategy uses
the 50/200 SMA cross as the PRIMARY signal, filtered by:
1. Weekly SMA(50) for trend confirmation (eliminates whipsaws against major trend)
2. Volume spike confirmation (institutional participation required)
3. ATR squeeze filter (only trade when volatility is rising = breakout potential)

WHY 1d: Natural trade frequency = 30-80/year, matching proven winners. 4h/6h overtrade.
WHY BOTH BULL AND BEAR: Death Cross shorts capture bear crashes. Golden Cross longs
capture bull rallies. Weekly SMA filter ensures we're not fighting the major trend.

TARGET: 30-100 total trades over 4 years. HARD MAX: 150.
Signal size: 0.30 (discrete).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_golden_death_cross_1w_sma200_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator for entry timing"""
    n = len(close)
    wr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest != lowest:
            wr[i] = -100 * (highest - close[i]) / (highest - lowest)
    
    return wr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load weekly HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly SMA50 for trend (aligned to daily)
    sma_1w = pd.Series(df_1w['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # Local daily indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # SMA for Golden/Death Cross
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # ATR for squeeze detection
    atr_30_ma = pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 250  # Need 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        
        if np.isnan(sma_200[i]) or np.isnan(sma_50[i]):
            continue
        
        if np.isnan(sma_1w_aligned[i]):
            continue
        
        if np.isnan(atr_30_ma[i]) or atr_30_ma[i] <= 1e-10:
            continue
        
        # === TREND DIRECTION (1w SMA50) ===
        price_above_1w_sma = close[i] > sma_1w_aligned[i]
        
        # === GOLDEN/DEATH CROSS detection ===
        # Need previous bar values for cross detection
        prev_sma_50 = sma_50[i - 1] if i > 0 else sma_50[i]
        prev_sma_200 = sma_200[i - 1] if i > 0 else sma_200[i]
        prev_close = close[i - 1] if i > 0 else close[i]
        
        golden_cross = (prev_sma_50 <= prev_sma_200) and (sma_50[i] > sma_200[i])
        death_cross = (prev_sma_50 >= prev_sma_200) and (sma_50[i] < sma_200[i])
        
        # === ATR SQUEEZE (volatility breakout filter) ===
        # Trade when ATR is expanding (not in compression)
        atr_expanding = atr_14[i] > atr_30_ma[i] * 0.95
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 1.0
        vol_spike = vol_ratio > 1.3
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Golden Cross + above weekly SMA ===
            if golden_cross and price_above_1w_sma:
                if vol_spike:  # Volume confirmation
                    desired_signal = SIZE
            
            # === SHORT: Death Cross + below weekly SMA ===
            if death_cross and not price_above_1w_sma:
                if vol_spike:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (1.5 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLDING PERIOD EXIT (5 days minimum) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 5:
            # Exit if price reverts to SMA50
            if position_side > 0 and close[i] < sma_50[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] > sma_50[i]:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = low[i] - 1.5 * entry_atr
                else:
                    stop_price = high[i] + 1.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals