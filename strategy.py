#!/usr/bin/env python3
"""
Experiment #1099: 4h Primary + 1d HTF — KAMA Adaptive Trend with RSI Pullback

Hypothesis: After analyzing 796+ failed experiments, key insights for 4h timeframe:
1. KAMA (Kaufman Adaptive Moving Average) adapts to market noise better than HMA/EMA
2. Complex regime filters (Choppiness, CRSI) kill trade frequency → 0 trades
3. SIMPLER entry conditions generate more trades while maintaining quality
4. 1d HMA provides clean macro trend filter without over-complication
5. RSI thresholds 35/65 (not extreme 20/80) ensure adequate trade frequency
6. Volume confirmation on breakouts reduces false signals
7. Position size 0.30 base with 2.5x ATR trailing stop

Why this should beat Sharpe=0.612 (current best):
- KAMA adapts ER (Efficiency Ratio) to reduce whipsaw in choppy markets
- 1d HMA macro filter prevents counter-trend trades in strong trends
- Volume filter confirms genuine breakouts vs fakeouts
- Loose RSI thresholds ensure 30-60 trades/year on 4h timeframe
- Proven pattern: KAMA + RSI + ATR worked on ETH (Sharpe +0.755 in research)

Timeframe: 4h (primary)
HTF: 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.30 base, 0.15 reduced (discrete levels)
Stoploss: 2.5x ATR trailing
Target: 30-60 trades/year, Sharpe > 0.612, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_volume_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    
    Adapts smoothing based on market efficiency (trend vs noise).
    ER (Efficiency Ratio) = |Net Change| / Sum of Absolute Changes
    High ER → trending → fast smoothing constant
    Low ER → choppy → slow smoothing constant
    
    Formula:
    1. ER = |close[i] - close[i-period]| / sum(|close[i-j] - close[i-j-1]|) for j in 0..period-1
    2. SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    3. KAMA[i] = KAMA[i-1] + SC * (close[i] - KAMA[i-1])
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + 1:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        net_change = abs(close[i] - close[i - period])
        sum_changes = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
    
    # Smoothing constants
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    # Calculate KAMA
    kama[period] = close[period]  # Initialize
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    Used for HTF macro trend filter.
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = int(period / 2)
    if half < 1:
        half = 1
    
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    diff = 2 * wma1 - wma2
    
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_volume_ma(volume, period=20):
    """Volume moving average for volume confirmation."""
    n = len(volume)
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=10, fast=2, slow=30)
    rsi_4h = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_ma = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_ma[i]) or vol_ma[i] <= 1e-10:
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === ADAPTIVE TREND (4h KAMA) ===
        # KAMA slope indicates trend direction
        kama_slope_bull = kama_4h[i] > kama_4h[i - 5] if i >= 5 else False
        kama_slope_bear = kama_4h[i] < kama_4h[i - 5] if i >= 5 else False
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama_4h[i]
        price_below_kama = close[i] < kama_4h[i]
        
        # === PULLBACK SIGNAL (4h RSI) ===
        # Loose thresholds to ensure adequate trade frequency
        rsi_oversold = rsi_4h[i] < 40.0
        rsi_overbought = rsi_4h[i] > 60.0
        
        # === VOLUME CONFIRMATION ===
        # Volume above average confirms genuine moves
        vol_above_avg = volume[i] > 1.2 * vol_ma[i]
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRY ===
        # Macro bull + KAMA bull + RSI pullback + volume confirmation
        if macro_bull and kama_slope_bull and price_above_kama:
            if rsi_oversold or (rsi_4h[i] < 50.0 and vol_above_avg):
                desired_signal = current_size
        
        # === SHORT ENTRY ===
        # Macro bear + KAMA bear + RSI pullback + volume confirmation
        elif macro_bear and kama_slope_bear and price_below_kama:
            if rsi_overbought or (rsi_4h[i] > 50.0 and vol_above_avg):
                desired_signal = -current_size
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bull and KAMA still rising
                if macro_bull and kama_slope_bull:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if macro still bear and KAMA still falling
                if macro_bear and kama_slope_bear:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses or RSI overbought
            if macro_bear or rsi_4h[i] > 70.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses or RSI oversold
            if macro_bull or rsi_4h[i] < 30.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals