#!/usr/bin/env python3
"""
Experiment #1524: 4h Primary + 12h/1d HTF — Simplified HMA Trend + RSI Pullback + Donchian

Hypothesis: Based on #1513 success (1d HMA+RSI+Donchian) and #1522 (12h HMA+CRSI), 
scaling to 4h with 12h HTF should work. Key lesson from 1134 failures:
1. Complex regime filters (CHOP+CRSI+ADX) = negative Sharpe or 0 trades
2. SIMPLER works: HTF trend + primary trend + RSI + Donchian (#1513, #1522 kept)
3. 4h timeframe naturally generates 20-50 trades/year (perfect for fee efficiency)
4. Use RSI(7) instead of RSI(14) for faster pullback detection on 4h
5. Fewer confluence requirements = more trades while maintaining quality

Design:
- 12h HMA(21) for macro trend direction (HTF filter)
- 1d HMA(21) for secondary confirmation (optional HTF)
- 4h HMA(21) for primary trend confirmation
- 4h RSI(7) for pullback entries (faster than RSI14, more responsive)
- 4h Donchian(20) breakout as momentum confirmation
- ATR(14) 2.5x trailing stop for risk management
- Position size 0.30 (discrete: 0.0, ±0.30)
- Target: 40-80 trades/train (4 years), 10-20 trades/test (15 months)

Timeframe: 4h (as required by experiment)
HTF: 12h + 1d (dual HTF for stronger trend confirmation)
Position Size: 0.30 (discrete levels to minimize fee churn)
Target: Sharpe > 0.618 (beat current best), DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi7_donchian_12h1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            if np.any(np.isnan(data[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=7):
    """Relative Strength Index - faster period for 4h"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 12h HMA for macro trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for secondary confirmation
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 4h
    atr = calculate_atr(high, low, close, period=14)
    
    # Donchian channels
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # Appropriate size for 4h (40-80 trades/year target)
    
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
        if np.isnan(rsi[i]) or np.isnan(hma_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (12h HMA) - primary direction bias ===
        twelve_h_bull = close[i] > hma_12h_aligned[i]
        twelve_h_bear = close[i] < hma_12h_aligned[i]
        
        # === SECONDARY TREND (1d HMA) - confirmation ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA) - entry confirmation ===
        four_h_bull = close[i] > hma_4h[i]
        four_h_bear = close[i] < hma_4h[i]
        
        # === RSI PULLBACK - RSI(7) for faster signals ===
        # Long: RSI pulled back but not oversold (25-50)
        rsi_pullback_long = 25.0 <= rsi[i] <= 50.0
        # Short: RSI rallied but not overbought (50-75)
        rsi_pullback_short = 50.0 <= rsi[i] <= 75.0
        
        # === DONCHIAN MOMENTUM - price near channel bounds ===
        donchian_range = donchian_upper[i] - donchian_lower[i]
        if donchian_range > 1e-10:
            donchian_position = (close[i] - donchian_lower[i]) / donchian_range
        else:
            donchian_position = 0.5
        
        donchian_bull = donchian_position > 0.55  # price in upper half
        donchian_bear = donchian_position < 0.45  # price in lower half
        
        # === DESIRED SIGNAL - SIMPLIFIED FOR 4h ===
        desired_signal = 0.0
        
        # LONG: 12h bullish + 4h bullish + RSI pullback
        # Option 1: Strong trend (12h + 4h both bull) + RSI pullback
        if twelve_h_bull and four_h_bull and rsi_pullback_long:
            desired_signal = BASE_SIZE
        # Option 2: 12h bull + 4h bull + Donchian bull (momentum entry)
        elif twelve_h_bull and four_h_bull and donchian_bull:
            desired_signal = BASE_SIZE * 0.9
        # Option 3: 12h bull + 4h bull + RSI not overbought (looser)
        elif twelve_h_bull and four_h_bull and rsi[i] < 60.0:
            desired_signal = BASE_SIZE * 0.7
        # Option 4: 12h bull + 4h above HMA (fallback for trades)
        elif twelve_h_bull and four_h_bull:
            desired_signal = BASE_SIZE * 0.5
        
        # SHORT: 12h bearish + 4h bearish + RSI pullback
        # Option 1: Strong trend (12h + 4h both bear) + RSI pullback
        elif twelve_h_bear and four_h_bear and rsi_pullback_short:
            desired_signal = -BASE_SIZE
        # Option 2: 12h bear + 4h bear + Donchian bear (momentum entry)
        elif twelve_h_bear and four_h_bear and donchian_bear:
            desired_signal = -BASE_SIZE * 0.9
        # Option 3: 12h bear + 4h bear + RSI not oversold (looser)
        elif twelve_h_bear and four_h_bear and rsi[i] > 40.0:
            desired_signal = -BASE_SIZE * 0.7
        # Option 4: 12h bear + 4h below HMA (fallback for trades)
        elif twelve_h_bear and four_h_bear:
            desired_signal = -BASE_SIZE * 0.5
        
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
        if desired_signal >= BASE_SIZE * 0.8:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.6:
            final_signal = BASE_SIZE * 0.7
        elif desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.8:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.6:
            final_signal = -BASE_SIZE * 0.7
        elif desired_signal <= -BASE_SIZE * 0.4:
            final_signal = -BASE_SIZE * 0.5
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