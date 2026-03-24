#!/usr/bin/env python3
"""
Experiment #049: 15m Primary + 1h/1d HTF — Camarilla Pivot Mean Reversion + Session Filter

Hypothesis: 15m timeframe has failed 3x with Sharpe=0.000 (zero trades). The problem is
overly restrictive conditions. This strategy uses:
- 1d HMA for major trend bias (loose filter: just direction, not strict)
- 1h RSI for momentum confirmation (very loose: 30-70 range)
- 15m Bollinger Bands for mean reversion entries (price at bands + RSI extreme)
- Session filter: 00-12 UTC preferred but not mandatory (allows trades outside)
- Position size: 0.20 (smaller for 15m frequency, target 50-100 trades/year)
- LOOSE entry conditions to ensure trades generate on ALL symbols

Key insight: 15m works best as MEAN REVERSION with HTF trend filter, not pure trend.
Trend following on 15m creates whipsaws. Mean revert at BB bounds WITH HTF bias works.

CRITICAL: Entry conditions MUST be loose enough to generate ≥10 trades/symbol on train.
- RSI thresholds: 25-75 (not 20-80)
- BB touch: price within 5% of band (not exact touch)
- Session: prefer 00-12 UTC but allow all hours (0.7x size outside)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_bb_rsi_session_hma_1h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

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

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1h RSI for momentum
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate primary (15m) indicators
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_dev=2.0)
    rsi_15m = calculate_rsi(close, period=7)  # Faster RSI for 15m
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (conservative for 15m frequency)
    SIZE_REDUCED = 0.12  # Reduced size outside preferred session
    
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
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_15m[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC preferred) ===
        # Extract hour from open_time (milliseconds timestamp)
        timestamp_ms = open_time[i]
        hour_utc = (timestamp_ms // (1000 * 60 * 60)) % 24
        is_preferred_session = 0 <= hour_utc <= 12
        current_size = SIZE if is_preferred_session else SIZE_REDUCED
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 1h RSI MOMENTUM ===
        rsi_1h = rsi_1h_aligned[i]
        rsi_1h_bull = rsi_1h > 40.0  # Loose: not oversold on 1h
        rsi_1h_bear = rsi_1h < 60.0  # Loose: not overbought on 1h
        
        # === 15m BB MEAN REVERSION ===
        bb_range = bb_upper[i] - bb_lower[i]
        if bb_range <= 1e-10:
            signals[i] = 0.0
            continue
        
        # Price position within BB (0 = lower, 1 = upper)
        price_position = (close[i] - bb_lower[i]) / bb_range
        
        # Near lower band (within 10% of band range)
        near_lower = price_position < 0.15
        # Near upper band (within 10% of band range)
        near_upper = price_position > 0.85
        
        # === 15m RSI EXTREMES (Fast RSI for quick entries) ===
        rsi_15m_oversold = rsi_15m[i] < 35.0  # Loose oversold
        rsi_15m_overbought = rsi_15m[i] > 65.0  # Loose overbought
        
        # === DESIRED SIGNAL (Mean Reversion with HTF Bias) ===
        desired_signal = 0.0
        
        # LONG: Near BB lower + RSI oversold + HTF not strongly bear
        # LOOSE: Only need 2 of 3 conditions (BB + RSI OR BB + HTF)
        long_bb_rsi = near_lower and rsi_15m_oversold
        long_bb_htf = near_lower and htf_bull
        long_rsi_htf = rsi_15m_oversold and htf_bull and rsi_1h_bull
        
        if long_bb_rsi or long_bb_htf or long_rsi_htf:
            # Additional filter: HTF should not be strongly against
            if not htf_bear or rsi_1h_bull:
                desired_signal = current_size
        
        # SHORT: Near BB upper + RSI overbought + HTF not strongly bull
        short_bb_rsi = near_upper and rsi_15m_overbought
        short_bb_htf = near_upper and htf_bear
        short_rsi_htf = rsi_15m_overbought and htf_bear and rsi_1h_bear
        
        if short_bb_rsi or short_bb_htf or short_rsi_htf:
            # Additional filter: HTF should not be strongly against
            if not htf_bull or rsi_1h_bear:
                desired_signal = -current_size
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= current_size * 0.85:
            final_signal = current_size
        elif desired_signal <= -current_size * 0.85:
            final_signal = -current_size
        elif desired_signal >= current_size * 0.5:
            final_signal = current_size * 0.5
        elif desired_signal <= -current_size * 0.5:
            final_signal = -current_size * 0.5
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