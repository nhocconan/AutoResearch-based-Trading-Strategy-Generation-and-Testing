#!/usr/bin/env python3
"""
Experiment #030: 1h Primary + 4h/12h HTF — Regime-Adaptive with LOOSE Entries

Hypothesis: After 29 experiments, the #1 failure mode is 0 trades (25+ strategies).
Entry conditions are TOO STRICT. This strategy uses LOOSER thresholds while
maintaining quality via HTF filter + regime detection.

Key changes from failed attempts:
1. LOOSER RSI: 35/65 instead of 30/70 (triggers more often)
2. LOOSER volume: >0.5x avg instead of >1.0x (still filters dead periods)
3. Session filter ONLY for entries (8-20 UTC), exits anytime
4. Only 3 confluence filters (HTF trend + RSI + volume), not 4-5
5. Asymmetric sizing: 0.30 in trend regime, 0.20 in chop regime

Strategy Logic:
- HTF (4h) HMA(21) determines trend direction
- 12h HMA(21) confirms macro bias
- Choppiness Index: >50 = chop (smaller size 0.20), <50 = trend (larger size 0.30)
- Entry: RSI(14) <35 long or >65 short + volume >0.5x avg + session 8-20 UTC
- Exit: RSI crosses 50 (mean reversion) OR stoploss (2.5x ATR)

Why this should work:
- 4h HMA proven edge (current best strategy uses it)
- LOOSE thresholds ensure trades generate (critical lesson from 25 failures)
- Regime-adaptive sizing reduces risk in choppy markets
- 1h TF with HTF filter = ~40-60 trades/year (within target 30-80)

Target: Sharpe>0.3, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 1h (lower TF = more trades, but HTF filter controls frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_loose_rsi_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, span):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    double_wma_half = 2.0 * wma_half - wma_full
    hma = wma(double_wma_half, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """RSI - momentum with LOOSE thresholds for more trades"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - regime detection (simplified formula)"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    choppiness = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        high_low_range = highest_high - lowest_low
        
        if high_low_range < 1e-10:
            choppiness[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        choppiness[i] = 100.0 * (atr_sum / period) / high_low_range
    
    return choppiness

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30  # Larger size in trending regime
    SIZE_CHOP = 0.20   # Smaller size in choppy regime
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(choppiness[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        choppy_regime = choppiness[i] > 50.0  # Range market
        trend_regime = choppiness[i] <= 50.0  # Trending market
        
        # Position size based on regime
        current_size = SIZE_CHOP if choppy_regime else SIZE_TREND
        
        # === HTF TREND BIAS ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_12h_bull = close[i] > hma_12h_aligned[i]
        hma_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === VOLUME FILTER (LOOSE: >0.5x avg) ===
        volume_ok = volume[i] > 0.5 * vol_sma[i]
        
        # === SESSION FILTER (8-20 UTC for entries only) ===
        current_hour = get_hour_from_open_time(open_time[i])
        session_ok = 8 <= current_hour <= 20
        
        # === RSI SIGNALS (LOOSE: 35/65 instead of 30/70) ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        rsi_neutral_exit = 45.0 < rsi[i] < 55.0  # Exit when RSI mean-reverts
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG entry: HTF bull + RSI oversold + volume + session
        if hma_4h_bull and hma_12h_bull and rsi_oversold and volume_ok and session_ok:
            desired_signal = current_size
        
        # SHORT entry: HTF bear + RSI overbought + volume + session
        elif hma_4h_bear and hma_12h_bear and rsi_overbought and volume_ok and session_ok:
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
        
        # === EXIT ON RSI MEAN REVERSION (when in profit) ===
        if in_position and not stoploss_triggered:
            if position_side > 0 and rsi_neutral_exit and close[i] > entry_price:
                desired_signal = 0.0  # Take profit on long
            elif position_side < 0 and rsi_neutral_exit and close[i] < entry_price:
                desired_signal = 0.0  # Take profit on short
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_TREND * 0.85:
            final_signal = SIZE_TREND
        elif desired_signal >= SIZE_CHOP * 0.85:
            final_signal = SIZE_CHOP
        elif desired_signal <= -SIZE_TREND * 0.85:
            final_signal = -SIZE_TREND
        elif desired_signal <= -SIZE_CHOP * 0.85:
            final_signal = -SIZE_CHOP
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
    
    return signals