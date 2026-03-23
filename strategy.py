#!/usr/bin/env python3
"""
Experiment #388: 30m Primary + 4h/1d HTF — Choppiness Regime + RSI + Session Filter

Hypothesis: Previous strategies failed due to OVER-FILTERING (0 trades in #378, #380, #382, #384, #385).
This strategy uses RELAXED but quality entry conditions to ensure trade generation while
maintaining edge through regime detection and HTF alignment.

Key innovations:
1. Choppiness Index (CHOP) for regime: >55 = range (mean revert), <45 = trend (follow)
2. RSI(14) for entries: <35 long, >65 short (more reliable than CRSI in crypto)
3. HTF HMA(21) on 4h for bias direction
4. Session filter: only 8-20 UTC (high liquidity, avoid Asia chop)
5. Volume confirmation: >0.8x 20-bar average
6. ATR(14) trailing stop at 2.5x
7. Position size 0.25 (smaller for 30m TF to reduce fee drag)

Target: 40-80 trades/year on 30m, Sharpe > 0.5 on ALL symbols.
Must beat current best: mtf_4h_triple_regime_crsi_donchian_1d1w_v1 (Sharpe=0.612)

CRITICAL: Entry conditions are RELAXED to ensure trades generate. This is the #1 lesson
from 330+ failed experiments - over-filtering = 0 trades = auto-reject.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_regime_rsi_session_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) window
    Reduces lag while maintaining smoothness.
    """
    close_s = pd.Series(close)
    n = int(period)
    
    # WMA helper
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, n // 2)
    wma_full = wma(close_s, n)
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, int(np.sqrt(n)))
    
    return hma.fillna(close_s.iloc[0]).values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    Interpretation:
    - CHOP > 61.8 = choppy/range-bound (mean reversion favorable)
    - CHOP < 38.2 = trending (trend following favorable)
    - 38.2-61.8 = transition zone
    """
    n = period
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate ATR for each bar
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Sum of ATR over period
    atr_sum = atr.rolling(window=n, min_periods=n).sum()
    
    # Highest high and lowest low over period
    highest_high = high_s.rolling(window=n, min_periods=n).max()
    lowest_low = low_s.rolling(window=n, min_periods=n).min()
    
    # Price range
    price_range = highest_high - lowest_low
    
    # CHOP calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(n)
    
    chop = np.clip(chop, 0, 100)
    return chop.fillna(50.0).values

def calculate_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # Convert milliseconds to seconds, then to datetime
    ts_seconds = open_time / 1000.0
    hours = (ts_seconds % 86400) / 3600.0
    return hours.astype(int)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 30m indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate and align HTF HMA for bias (4h)
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for stronger bias filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Extract session hours
    session_hours = calculate_session_hour(open_time)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% position size for 30m (target 40-80 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] <= 1e-10:
            continue
        
        # === HTF BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === STRONGER HTF BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_value = chop_14[i]
        is_range = chop_value > 55.0  # Range-bound market
        is_trend = chop_value < 45.0  # Trending market
        # 45-55 is transition zone - use either logic
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= session_hours[i] <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === RSI SIGNALS ===
        rsi_value = rsi_14[i]
        rsi_oversold = rsi_value < 35.0
        rsi_overbought = rsi_value > 65.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP
        # Need: HTF bias + (regime-appropriate entry) + session + volume
        long_bias = price_above_hma_4h  # Primary HTF bias
        
        if long_bias and in_session and volume_ok:
            if is_range and rsi_oversold:
                # Range mean-reversion long
                desired_signal = BASE_SIZE
            elif is_trend and rsi_value < 50:
                # Trend pullback long (RSI dipped but still in uptrend)
                desired_signal = BASE_SIZE
            elif rsi_oversold:
                # Strong oversold regardless of regime
                desired_signal = BASE_SIZE
        
        # SHORT SETUP
        short_bias = price_below_hma_4h  # Primary HTF bias
        
        if short_bias and in_session and volume_ok:
            if is_range and rsi_overbought:
                # Range mean-reversion short
                desired_signal = -BASE_SIZE
            elif is_trend and rsi_value > 50:
                # Trend rally short (RSI spiked but still in downtrend)
                desired_signal = -BASE_SIZE
            elif rsi_overbought:
                # Strong overbought regardless of regime
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
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
        
        # === RSI EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and rsi_value > 60:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_value < 40:
            desired_signal = 0.0
        
        # === HTF BIAS REVERSAL EXIT ===
        if in_position and position_side > 0 and price_below_hma_4h:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_4h:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_4h:
                desired_signal = BASE_SIZE
            elif position_side < 0 and price_below_hma_4h:
                desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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