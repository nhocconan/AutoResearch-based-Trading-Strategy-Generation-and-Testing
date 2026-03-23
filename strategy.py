#!/usr/bin/env python3
"""
Experiment #1122: 12h Primary + 1d/1w HTF — Simplified Multi-TF Trend Following

Hypothesis: After analyzing 800+ failed experiments, key insights:
1. Complex regime-switching (Choppiness + CRSI) causes 0 trades or whipsaw
2. 12h timeframe naturally produces 20-50 trades/year — optimal frequency
3. SIMPLER multi-TF works better: 1w HMA (macro) + 1d HMA (confirmation) + 12h RSI (entry)
4. KAMA (Kaufman Adaptive MA) adapts to volatility better than EMA/HMA in chop
5. Loose RSI thresholds (35/65) ensure adequate trade frequency without overtrading
6. ATR trailing stop (2.5x) protects capital during 2022-style crashes

Why this should beat Sharpe=0.612 (current best):
- 1w HMA provides ultra-stable macro filter (less whipsaw than 1d)
- KAMA adapts to market regime automatically (no manual Choppiness calc needed)
- 12h RSI pullback entries catch dips in uptrends, rallies in downtrends
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn
- Proven pattern from research: HMA+RSI+ATR on SOL achieved Sharpe +0.879

Timeframe: 12h (primary)
HTF: 1d and 1w — loaded ONCE before loop using mtf_data helper
Position Size: 0.25 base, 0.30 strong signal (discrete levels)
Stoploss: 2.5x ATR trailing
Target: 25-50 trades/year per symbol, Sharpe > 0.612, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_rsi_1d1w_hma_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average — adapts to market noise.
    
    Formula:
    1. Efficiency Ratio (ER) = |Close - Close_n| / Sum(|Close_i - Close_i-1|)
    2. Smoothing Constant (SC) = [ER * (fast_sc - slow_sc) + slow_sc]^2
    3. KAMA = KAMA_prev + SC * (Close - KAMA_prev)
    
    ER near 1 = trending (fast response), ER near 0 = choppy (slow response)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    numerator = np.abs(close - np.roll(close, er_period))
    numerator[:er_period] = np.nan
    
    denominator = np.zeros(n)
    for i in range(er_period, n):
        denominator[i] = np.sum(np.abs(close[i-er_period+1:i+1] - np.roll(close[i-er_period+1:i+1], 1))[1:])
    
    denominator[denominator == 0] = 1e-10
    er = numerator / denominator
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    
    # Initialize KAMA with SMA
    kama[er_period] = np.nanmean(close[:er_period+1])
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        if np.isnan(kama[i-1]):
            kama[i] = kama[er_period]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
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
    
    half = max(1, int(period / 2))
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    diff = 2 * wma1 - wma2
    sqrt_period = max(1, int(np.sqrt(period)))
    hma = wma(diff, sqrt_period)
    return hma

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for ultra-long-term macro trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d HMA for medium-term trend confirmation
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    kama_12h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi_12h = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_12h[i]) or np.isnan(atr[i]) or np.isnan(kama_12h[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1w HMA) — Ultra-long-term bias ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === CONFIRMATION (1d HMA) — Medium-term alignment ===
        confirm_bull = close[i] > hma_1d_aligned[i]
        confirm_bear = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND (12h) — Adaptive trend direction ===
        kama_bull = close[i] > kama_12h[i]
        kama_bear = close[i] < kama_12h[i]
        
        # === PULLBACK SIGNAL (12h RSI) — Entry timing ===
        # Loose thresholds for adequate trade frequency
        rsi_oversold = rsi_12h[i] < 40.0
        rsi_overbought = rsi_12h[i] > 60.0
        rsi_neutral = 35.0 < rsi_12h[i] < 65.0
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRY ===
        # All 3 timeframes aligned bull + RSI pullback
        if macro_bull and confirm_bull and kama_bull:
            if rsi_oversold:
                current_size = STRONG_SIZE  # Strong signal on deep pullback
            elif rsi_neutral and rsi_12h[i] < 50.0:
                current_size = BASE_SIZE
            else:
                current_size = 0.0
            
            if current_size > 0:
                desired_signal = current_size
        
        # === SHORT ENTRY ===
        # All 3 timeframes aligned bear + RSI pullback
        elif macro_bear and confirm_bear and kama_bear:
            if rsi_overbought:
                current_size = STRONG_SIZE  # Strong signal on deep rally
            elif rsi_neutral and rsi_12h[i] > 50.0:
                current_size = BASE_SIZE
            else:
                current_size = 0.0
            
            if current_size > 0:
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
                # Hold long if macro and confirmation still bull
                if macro_bull and confirm_bull:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if macro and confirmation still bear
                if macro_bear and confirm_bear:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses or KAMA flips
            if not macro_bull or not kama_bull:
                desired_signal = 0.0
            # Exit on RSI overbought
            elif rsi_12h[i] > 75.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses or KAMA flips
            if not macro_bear or not kama_bear:
                desired_signal = 0.0
            # Exit on RSI oversold
            elif rsi_12h[i] < 25.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= STRONG_SIZE * 0.8:
                desired_signal = STRONG_SIZE
            elif desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            else:
                desired_signal = BASE_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -STRONG_SIZE * 0.8:
                desired_signal = -STRONG_SIZE
            elif desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = -BASE_SIZE * 0.5
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