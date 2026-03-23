#!/usr/bin/env python3
"""
Experiment #1160: 1h Primary + 4h/12h HTF — Simplified Trend Pullback Strategy

Hypothesis: Previous 1h strategies (#1150, #1155, #1158) failed with 0 trades due to
overly strict confluence filters. This strategy SIMPLIFIES entry logic while keeping
HTF trend filter for direction.

Key changes from failed experiments:
1. LOOSEN RSI thresholds (30-50 for long, 50-70 for short) — extremes cause 0 trades
2. Reduce volume filter to 1.2x (not 1.5x) — still confirms but allows more trades
3. Remove session filter — was killing trades on BTC/ETH which trade 24/7
4. Single HTF trend filter (12h HMA) — not multiple conflicting HTF signals
5. Position size 0.25 (conservative for 1h TF)

Structure:
- 12h HMA(21) = macro trend direction (loaded ONCE via mtf_data)
- 4h RSI(14) = pullback detection in direction of 12h trend
- 1h volume > 1.2x SMA20 = confirmation
- 1h ATR(14) 2.5x trailing stop = risk management

Target: 40-80 trades/year on 1h (optimal for fee drag at this TF)
Expected: Sharpe > 0.612 (beat current best), DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h12h_simplified_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """
    Relative Strength Index — momentum oscillator.
    RSI = 100 - 100/(1 + RS), RS = avg_gain/avg_loss
    """
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 1e-10
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[:period] = np.nan
    
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

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume."""
    n = len(volume)
    vol_sma = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        vol_sma[i] = np.mean(volume[i - period + 1:i + 1])
    
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 12h HMA for macro trend filter
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 4h RSI for pullback detection
    rsi_4h_raw = calculate_rsi(df_4h['close'].values, period=14)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_raw)
    
    # Calculate primary (1h) indicators
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    rsi_1h = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(vol_sma[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or hma_12h_aligned[i] <= 1e-10:
            continue
        if np.isnan(rsi_4h_aligned[i]) or np.isnan(rsi_1h[i]):
            continue
        if vol_sma[i] <= 1e-10 or atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (12h HMA) ===
        # Simple: price above HMA = bull, below = bear
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === PULLBACK DETECTION (4h RSI) ===
        # Long: RSI pulled back to 35-50 range (not oversold, just resting)
        # Short: RSI pulled back to 50-65 range (not overbought, just resting)
        rsi_4h = rsi_4h_aligned[i]
        pullback_long = 35.0 <= rsi_4h <= 55.0
        pullback_short = 45.0 <= rsi_4h <= 65.0
        
        # === VOLUME CONFIRMATION (1h) ===
        # Volume > 1.2x average confirms interest (looser than 1.5x to allow more trades)
        volume_ok = volume[i] > 1.2 * vol_sma[i]
        
        # === VOLATILITY FILTER ===
        # Skip if ATR is extremely low (dead market) or extremely high (panic)
        atr_ratio = atr[i] / np.nanmean(atr[max(0, i-100):i]) if i > 100 else 1.0
        vol_ok = 0.5 < atr_ratio < 3.0
        
        # === ENTRY CONDITIONS (SIMPLIFIED) ===
        desired_signal = 0.0
        
        # LONG: Macro bull + RSI pullback + volume confirmation
        if macro_bull and pullback_long and volume_ok and vol_ok:
            desired_signal = BASE_SIZE
        
        # SHORT: Macro bear + RSI pullback + volume confirmation
        elif macro_bear and pullback_short and volume_ok and vol_ok:
            desired_signal = -BASE_SIZE
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and macro_bear:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bull:
            desired_signal = 0.0
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                if macro_bull and vol_ok:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                if macro_bear and vol_ok:
                    desired_signal = -BASE_SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
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