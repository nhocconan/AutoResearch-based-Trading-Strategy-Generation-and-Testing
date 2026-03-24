#!/usr/bin/env python3
"""
Experiment #124: 4h Primary + 12h HTF — Fisher Transform Reversals + HMA Trend

Hypothesis: After 100+ failed experiments, the pattern is clear:
- KAMA crossovers are too slow and generate 0 trades in ranging markets
- RSI filters (even loose 25/75) combined with other conditions kill trade frequency
- Fisher Transform catches reversals better than RSI in bear/range markets (2022, 2025)
- 4h timeframe with 12h HTF bias worked for SOL (+0.879) — replicate with Fisher

This strategy uses MINIMAL but effective filters:
1. 12h HMA(21) = major trend bias (price above/below)
2. 4h Fisher Transform(9) = entry trigger (crosses -1.5 long, +1.5 short)
3. NO RSI filter (removes one failure point)
4. ATR trailing stoploss (2.5x) for risk management
5. Volume confirmation optional (only require > 0.8x avg to avoid dead zones)

Key design choices:
- Timeframe: 4h (proven, 20-50 trades/year target)
- HTF: 12h for trend bias (more responsive than 1d, less noise than 4h)
- Fisher Transform: period=9, normalized to [-2, +2], catches reversals early
- Position size: 0.28 (28% of capital, conservative for 4h)
- Stoploss: 2.5x ATR trailing (proven in baseline)
- NO complex regime filters (Choppiness caused 0 trades in #120, #121)

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_hma_reversal_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(prices, period=9):
    """
    Ehlers Fisher Transform
    Converts price to a Gaussian normal distribution for clearer reversal signals
    Formula: Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = normalized price
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(prices)
    if n < period + 5:
        return np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    # Calculate highest high and lowest low over period
    for i in range(period, n):
        hh = np.max(prices[i-period+1:i+1])
        ll = np.min(prices[i-period+1:i+1])
        
        if hh == ll:
            continue
        
        # Normalize price to range [-1, +1]
        x = 2.0 * (prices[i] - ll) / (hh - ll) - 1.0
        
        # Clamp to avoid division by zero in log
        x = np.clip(x, -0.999, 0.999)
        
        # Fisher Transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
    
    return fisher

def calculate_hma(close, period=21):
    """
    Hull Moving Average
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    More responsive than EMA with less lag
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    hma = np.zeros(n)
    hma[:] = np.nan
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    # Calculate WMA for half period
    wma_half = np.zeros(n)
    for i in range(half - 1, n):
        weights = np.arange(1, half + 1)
        wma_half[i] = np.sum(close[i-half+1:i+1] * weights) / np.sum(weights)
    
    # Calculate WMA for full period
    wma_full = np.zeros(n)
    for i in range(period - 1, n):
        weights = np.arange(1, period + 1)
        wma_full[i] = np.sum(close[i-period+1:i+1] * weights) / np.sum(weights)
    
    # Calculate raw HMA
    raw_hma = 2.0 * wma_half - wma_full
    
    # Final WMA on raw HMA with sqrt(period)
    for i in range(period - 1 + sqrt_n - 1, n):
        start = i - sqrt_n + 1
        weights = np.arange(1, sqrt_n + 1)
        hma[i] = np.sum(raw_hma[start:i+1] * weights) / np.sum(weights)
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume for volume confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = np.zeros(n)
    vol_sma[:] = np.nan
    for i in range(period - 1, n):
        vol_sma[i] = np.mean(volume[i-period+1:i+1])
    
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for major trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    fisher = calculate_fisher_transform(close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative for 4h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track Fisher crosses to avoid repeated signals
    prev_fisher = 0.0
    fisher_cross_up = False
    fisher_cross_down = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h HMA) ===
        # Simple: is price above or below 12h HMA?
        htf_bull = close[i] > hma_12h_aligned[i]
        htf_bear = close[i] < hma_12h_aligned[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNAL ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_up = (prev_fisher < -1.5 and fisher[i] >= -1.5)
        fisher_cross_down = (prev_fisher > 1.5 and fisher[i] <= 1.5)
        
        # === VOLUME CONFIRMATION (loose - avoid dead zones only) ===
        vol_ok = True
        if not np.isnan(vol_sma[i]) and vol_sma[i] > 1e-10:
            vol_ok = volume[i] > 0.5 * vol_sma[i]  # At least 50% of avg volume
        
        # === DESIRED SIGNAL ===
        # LONG: 12h bull + Fisher cross up + volume ok
        # SHORT: 12h bear + Fisher cross down + volume ok
        desired_signal = 0.0
        
        if htf_bull and fisher_cross_up and vol_ok:
            desired_signal = SIZE
        elif htf_bear and fisher_cross_down and vol_ok:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
        prev_fisher = fisher[i]
    
    return signals