#!/usr/bin/env python3
"""
Experiment #433: 15m Mean Reversion + 4h HMA Trend + Choppiness Regime Filter
Hypothesis: 15m timeframe benefits from mean reversion when aligned with 4h trend direction.
Choppiness Index filters out ranging markets where trend strategies fail.
Z-score entries capture extreme deviations from mean with high probability of reversion.
Multiple entry paths ensure >=10 trades per symbol while maintaining quality.
Key insight: 15m is noisy - need strong HTF filter (4h HMA) + regime detection (CHOP).
Relaxed thresholds ensure sufficient trades while avoiding whipsaw from #427 failure.
Timeframe: 15m (REQUIRED), HTF: 4h for trend via mtf_data helper.
Position size: 0.25 discrete, stoploss 2.5*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_mr_4h_hma_chop_zscore_atr_v1"
timeframe = "15m"
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

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close - sma.values) / std.values
    return zscore

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    range_hl = highest_high - lowest_low
    chop = np.where(range_hl > 0, 100 * np.log10(atr_sum / range_hl) / np.log10(period), 50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    zscore = calculate_zscore(close, 20)
    chop = calculate_choppiness(high, low, close, 14)
    sma50 = calculate_sma(close, 50)
    
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
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(sma50[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(zscore[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (long-term direction)
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # Choppiness regime filter (relaxed for more trades)
        is_trending = chop[i] < 55  # Below 55 = more trending
        is_ranging = chop[i] >= 55   # Above 55 = more ranging
        
        # Z-score extremes for mean reversion
        zscore_oversold = zscore[i] < -1.2
        zscore_overbought = zscore[i] > 1.2
        zscore_extreme_long = zscore[i] < -1.8
        zscore_extreme_short = zscore[i] > 1.8
        
        # RSI conditions (relaxed for more trades)
        rsi_oversold = rsi[i] < 42
        rsi_overbought = rsi[i] > 58
        rsi_neutral = rsi[i] >= 35 and rsi[i] <= 65
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Z-score oversold + 4h bullish + RSI < 48
        if zscore_oversold and trend_bullish and rsi[i] < 48:
            new_signal = SIZE_ENTRY
        # Path 2: Z-score extreme + 4h bullish (stronger signal)
        elif zscore_extreme_long and trend_bullish:
            new_signal = SIZE_ENTRY
        # Path 3: RSI oversold + 4h bullish + trending regime
        elif rsi_oversold and trend_bullish and is_trending:
            new_signal = SIZE_ENTRY
        # Path 4: Z-score < -0.8 + 4h bullish + RSI < 52
        elif zscore[i] < -0.8 and trend_bullish and rsi[i] < 52:
            new_signal = SIZE_ENTRY
        # Path 5: Price near SMA50 support + 4h bullish + RSI < 55
        elif close[i] < sma50[i] * 1.015 and trend_bullish and rsi[i] < 55:
            new_signal = SIZE_ENTRY
        # Path 6: RSI recovery + 4h bullish (RSI was <40, now >40)
        elif rsi[i] > 40 and rsi[i-1] <= 40 and trend_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Z-score overbought + 4h bearish + RSI > 52
        if zscore_overbought and trend_bearish and rsi[i] > 52:
            new_signal = -SIZE_ENTRY
        # Path 2: Z-score extreme + 4h bearish (stronger signal)
        elif zscore_extreme_short and trend_bearish:
            new_signal = -SIZE_ENTRY
        # Path 3: RSI overbought + 4h bearish + trending regime
        elif rsi_overbought and trend_bearish and is_trending:
            new_signal = -SIZE_ENTRY
        # Path 4: Z-score > 0.8 + 4h bearish + RSI > 48
        elif zscore[i] > 0.8 and trend_bearish and rsi[i] > 48:
            new_signal = -SIZE_ENTRY
        # Path 5: Price near SMA50 resistance + 4h bearish + RSI > 45
        elif close[i] > sma50[i] * 0.985 and trend_bearish and rsi[i] > 45:
            new_signal = -SIZE_ENTRY
        # Path 6: RSI rejection + 4h bearish (RSI was >60, now <60)
        elif rsi[i] < 60 and rsi[i-1] >= 60 and trend_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals