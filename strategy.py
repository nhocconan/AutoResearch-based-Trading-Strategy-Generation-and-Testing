#!/usr/bin/env python3
"""
Experiment #390: 1h Primary + 4h/12h HTF — Simplified MTF Trend + RSI Pullback

Hypothesis: Previous 1h/30m strategies failed (Sharpe=0.000) due to OVER-FILTERING.
This strategy uses SIMPLIFIED confluence with relaxed thresholds to ensure trade generation:

1. HTF (4h + 12h) HMA for DIRECTION only — dual HTF confirms trend bias
2. 1h RSI(14) for ENTRY TIMING — relaxed thresholds: <40 long, >60 short (not extreme 10/90)
3. Volume filter: >0.7x 20-bar avg (permissive, not restrictive)
4. Session filter for ENTRIES ONLY (8-20 UTC) — exits can happen anytime
5. ATR(14) stoploss at 2.5x — mandatory risk management
6. Position size: 0.25 (smaller for 1h to control fee drag and DD)

Key insight: Use HTF for SIGNAL DIRECTION, 1h only for ENTRY TIMING.
This gives HTF trade frequency (~40-60/year) with 1h execution precision.

Target: 40-80 trades/year, Sharpe > 0.5 on ALL symbols (BTC/ETH/SOL individually).
Must beat current best: mtf_4h_triple_regime_crsi_donchian_1d1w_v1 (Sharpe=0.612)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h12h_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    More responsive than EMA with less lag.
    """
    close_s = pd.Series(close)
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    hull_raw = 2.0 * wma_half - wma_full
    hma = hull_raw.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
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

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = volume / (vol_avg.values + 1e-10)
    return np.nan_to_num(ratio, nan=1.0)

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    return (open_time // 3600000) % 24

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
    
    # Calculate 1h indicators (primary timeframe)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Calculate and align HTF HMA for bias (4h + 12h)
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% position size for 1h (target 40-80 trades/year)
    
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
        if np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        
        # Extract hour for session filter
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20  # UTC 8-20 (London + NY overlap)
        
        # === HTF BIAS (4h + 12h HMA confluence) ===
        # Both HTF must agree for strong bias
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # Strong bullish: both 4h and 12h HMA below price
        strong_bull = price_above_hma_4h and price_above_hma_12h
        # Strong bearish: both 4h and 12h HMA above price
        strong_bear = price_below_hma_4h and price_below_hma_12h
        
        # === VOLUME FILTER (permissive) ===
        volume_ok = vol_ratio[i] > 0.7  # At least 70% of average volume
        
        # === RSI ENTRY TIMING (relaxed thresholds) ===
        # Long: RSI < 40 (pullback in uptrend)
        # Short: RSI > 60 (rally in downtrend)
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: Strong HTF bull + RSI pullback + volume + session
        if strong_bull and rsi_oversold and volume_ok and in_session:
            desired_signal = BASE_SIZE
        
        # SHORT ENTRY: Strong HTF bear + RSI rally + volume + session
        if strong_bear and rsi_overbought and volume_ok and in_session:
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
        # Exit long when RSI > 65 (overbought)
        # Exit short when RSI < 35 (oversold)
        if in_position and position_side > 0 and rsi_14[i] > 65:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 35:
            desired_signal = 0.0
        
        # === HTF EXIT (bias reversal) ===
        # Exit long if 12h HMA flips bearish
        # Exit short if 12h HMA flips bullish
        if in_position and position_side > 0 and price_below_hma_12h:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_12h:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        # If already in position and no exit trigger, hold the position
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and strong_bull:
                desired_signal = BASE_SIZE
            elif position_side < 0 and strong_bear:
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