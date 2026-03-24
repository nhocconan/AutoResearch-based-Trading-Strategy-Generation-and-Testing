#!/usr/bin/env python3
"""
Experiment #1675: 1h Primary + 4h/1d HTF — Simplified RSI Mean Reversion with Regime-Adjusted Sizing

Hypothesis: Previous 1h/30m strategies failed (Sharpe=0.000) due to OVER-FILTERING.
This strategy uses PROVEN patterns with SIMPLER logic to ensure trade generation:
- 4h HMA for trend direction (proven in #1664 with +16.4% return)
- 1h RSI(14) for entry timing (simpler than CRSI, more reliable signals)
- Choppiness Index adjusts SIZE not blocks trades (critical for trade generation)
- 1d HMA for broader bias confirmation
- LOOSE RSI thresholds (25/75) to ensure trades on all symbols
- Fallback logic: if no regime signal, trade with 1d trend only

Key differences from failed 1h attempts:
1. NO session filter (8-20 UTC was too restrictive)
2. NO volume filter (eliminated too many valid signals)
3. Choppiness adjusts size (0.30 trend / 0.20 range) not entry block
4. RSI 25/75 thresholds (not 30/70) for more signals
5. Fallback: always trade with 1d trend if RSI extreme

Entry Logic:
- RSI < 25 + 4h bullish → long 0.30 (trend) or 0.20 (range)
- RSI > 75 + 4h bearish → short 0.30 (trend) or 0.20 (range)
- Fallback: RSI < 30 + 1d bullish → long 0.15 (reduced size)
- Fallback: RSI > 70 + 1d bearish → short 0.15 (reduced size)

Risk: 2.5x ATR trailing stop, discrete signal levels (0.0, ±0.15, ±0.20, ±0.30)
Target: Sharpe > 0.618, trades > 30/symbol train, > 3/symbol test, DD > -40%
Trade Frequency: Target 40-80/year (strict enough for 1h, loose enough for signals)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_hma_4h1d_chop_size_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """
    Relative Strength Index (RSI)
    RSI = 100 - (100 / (1 + RS))
    RS = average_gain / average_loss
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i-1] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i-1] / avg_loss[i-1]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_hma(close, period=21):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 55 = choppy/range (use smaller size)
    CHOP < 45 = trending (use larger size)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], abs(high[j] - prev_close), abs(low[j] - prev_close))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
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
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for broader trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing levels (discrete to minimize fee churn)
    SIZE_TREND = 0.30      # Full size in trending regime
    SIZE_RANGE = 0.20      # Reduced size in choppy regime
    SIZE_FALLBACK = 0.15   # Minimum size for fallback signals
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # Select base size based on regime
        if is_trending:
            base_size = SIZE_TREND
        else:
            base_size = SIZE_RANGE
        
        # === HTF TREND BIAS ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY SIGNAL (RSI extremes + 4h trend) ===
        desired_signal = 0.0
        
        # Long: RSI < 25 (oversold) + 4h bullish bias
        if rsi[i] < 25.0:
            if hma_4h_bull:
                desired_signal = base_size
            elif hma_1d_bull:
                # Fallback: 1d bullish is enough for reduced size
                desired_signal = SIZE_FALLBACK
        
        # Short: RSI > 75 (overbought) + 4h bearish bias
        elif rsi[i] > 75.0:
            if hma_4h_bear:
                desired_signal = -base_size
            elif hma_1d_bear:
                # Fallback: 1d bearish is enough for reduced size
                desired_signal = -SIZE_FALLBACK
        
        # === FALLBACK SIGNAL (ensure trades generate) ===
        # If no primary signal but 1d trend is strong, take smaller position
        if desired_signal == 0.0:
            # Very oversold + 1d bull
            if rsi[i] < 30.0 and hma_1d_bull:
                desired_signal = SIZE_FALLBACK
            # Very overbought + 1d bear
            elif rsi[i] > 70.0 and hma_1d_bear:
                desired_signal = -SIZE_FALLBACK
        
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
        if desired_signal >= SIZE_TREND * 0.85:
            final_signal = SIZE_TREND
        elif desired_signal <= -SIZE_TREND * 0.85:
            final_signal = -SIZE_TREND
        elif desired_signal >= SIZE_RANGE * 0.85:
            final_signal = SIZE_RANGE
        elif desired_signal <= -SIZE_RANGE * 0.85:
            final_signal = -SIZE_RANGE
        elif desired_signal >= SIZE_FALLBACK * 0.85:
            final_signal = SIZE_FALLBACK
        elif desired_signal <= -SIZE_FALLBACK * 0.85:
            final_signal = -SIZE_FALLBACK
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
                # Position flip
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
    
    return signals