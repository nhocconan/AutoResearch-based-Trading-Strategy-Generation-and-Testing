#!/usr/bin/env python3
"""
Experiment #363: 1h Regime-Adaptive Strategy with 4h HMA Trend + Choppiness + Bollinger
Hypothesis: 1h timeframe balances signal frequency and noise better than 15m/30m (which failed 
with Sharpe -5 to -8) and 4h/12h (fewer opportunities). Key innovation: detect market regime 
using Choppiness Index and apply different entry logic per regime to handle both 2022 crash 
(whipsaw) and 2025 bear market (trend failure):
- Trending regime (CHOP < 45): Follow 4h HMA direction with Donchian breakouts
- Ranging regime (CHOP > 55): Mean revert at Bollinger extremes with RSI confirmation
- LOOSE thresholds to ensure 10+ trades per symbol (learned from failures)

Position sizing: 0.25 entry, 0.125 half, stoploss at 2.0*ATR. Discrete levels to minimize fees.
Target: Beat Sharpe=0.499 with 50-100 trades across train+test, DD < -30%.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_chop_4h_hma_bollinger_rsi_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market, CHOP < 38.2 = trending market
    Using relaxed thresholds (45/55) for more regime transitions.
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        atr_sum = 0.0
        highest_high = high[i - period + 1]
        lowest_low = low[i - period + 1]
        
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
            highest_high = max(highest_high, high[j])
            lowest_low = min(lowest_low, low[j])
        
        price_range = highest_high - lowest_low
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - MANDATORY)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align 4h HMA to 1h timeframe (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (LOOSE - just direction, not strict filter)
        trend_bullish = not np.isnan(hma_4h_aligned[i]) and close[i] > hma_4h_aligned[i]
        trend_bearish = not np.isnan(hma_4h_aligned[i]) and close[i] < hma_4h_aligned[i]
        
        # Regime detection (relaxed thresholds for more trades)
        is_trending = chop[i] < 45.0
        is_ranging = chop[i] > 55.0
        is_neutral = not is_trending and not is_ranging
        
        new_signal = 0.0
        
        # === TRENDING REGIME (CHOP < 45) ===
        if is_trending:
            # Long: 4h bullish OR Donchian breakout + RSI ok
            if (trend_bullish or close[i] > donchian_upper[i-1]) and rsi[i] < 75:
                new_signal = SIZE_ENTRY
            # Short: 4h bearish OR Donchian breakdown + RSI ok
            elif (trend_bearish or close[i] < donchian_lower[i-1]) and rsi[i] > 25:
                new_signal = -SIZE_ENTRY
        
        # === RANGING REGIME (CHOP > 55) ===
        elif is_ranging:
            # Long: Price at lower BB + RSI oversold (mean reversion)
            if close[i] < bb_lower[i] and rsi[i] < 40:
                new_signal = SIZE_ENTRY
            # Short: Price at upper BB + RSI overbought (mean reversion)
            elif close[i] > bb_upper[i] and rsi[i] > 60:
                new_signal = -SIZE_ENTRY
        
        # === NEUTRAL/TRANSITIONAL REGIME ===
        else:
            # Hybrid: trend bias + momentum
            if trend_bullish and rsi[i] > 40 and rsi[i] < 70:
                new_signal = SIZE_ENTRY
            elif trend_bearish and rsi[i] > 30 and rsi[i] < 60:
                new_signal = -SIZE_ENTRY
            # Fallback: Bollinger mean reversion
            elif close[i] < bb_lower[i] and rsi[i] < 35:
                new_signal = SIZE_ENTRY
            elif close[i] > bb_upper[i] and rsi[i] > 65:
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6 - MANDATORY) ===
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                risk = 2.0 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals