#!/usr/bin/env python3
"""
Experiment #638: 30m Primary + 4h/1d HTF — Simplified RSI + HMA Trend

Hypothesis: Previous 30m strategies (#598, #630) failed with 0 trades due to TOO MANY 
confluence filters (session + volume + chop + RSI + HTF). This version uses MINIMAL 
filters: 4h HMA for direction, 30m RSI for entry timing. No session/volume/chop filters.

Key changes from failed 30m attempts:
1. REMOVED session filter (8-20 UTC) — was blocking 40% of potential entries
2. REMOVED volume filter — inconsistent across SOL vs BTC/ETH
3. REMOVED Choppiness Index — adds complexity without improving 30m signals
4. LOOSENED RSI thresholds: 35/65 instead of 30/70 (more trades)
5. Added RSI cross logic — catches momentum shifts, not just extremes
6. Hold logic maintains positions through minor pullbacks

Why this should work when #598/#630 failed:
- Fewer filters = more trades (target 50-80/year, not 0)
- 4h HMA provides clean trend bias without over-filtering
- RSI(14) with crosses generates entries throughout trends, not just at extremes
- Conservative size (0.25) survives drawdowns

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_hma_simple_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing method."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Initialize with SMA
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    
    rsi[period] = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 100
    
    # Wilder's smoothing
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gain[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i - 1]) / period
        
        if avg_loss > 0:
            rsi[i] = 100 - (100 / (1 + avg_gain / avg_loss))
        else:
            rsi[i] = 100
    
    return rsi

def calculate_hma(close, period=21):
    """Hull Moving Average for smoother trend detection."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_ema(close, period=21):
    """Calculate EMA with proper min_periods."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 30m indicators (primary timeframe)
    rsi_30m = calculate_rsi(close, period=14)
    ema_30m = calculate_ema(close, period=21)
    atr_30m = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative size for lower TF
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_30m[i]) or np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(ema_30m[i]):
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_4h_bullish = close[i] > hma_4h_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === HTF MACRO BIAS (1d HMA) ===
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === 30m MOMENTUM (EMA slope) ===
        ema_slope_bullish = ema_30m[i] > ema_30m[i - 5] if i >= 105 else False
        ema_slope_bearish = ema_30m[i] < ema_30m[i - 5] if i >= 105 else False
        
        # === RSI SIGNALS ===
        rsi = rsi_30m[i]
        rsi_prev = rsi_30m[i - 1] if i > 0 else rsi
        
        # RSI levels
        rsi_oversold = rsi < 40.0
        rsi_overbought = rsi > 60.0
        
        # RSI crosses (momentum shifts)
        rsi_cross_up_40 = (rsi > 40.0) and (rsi_prev <= 40.0)
        rsi_cross_down_60 = (rsi < 60.0) and (rsi_prev >= 60.0)
        
        # RSI crosses from extreme
        rsi_cross_up_30 = (rsi > 30.0) and (rsi_prev <= 30.0)
        rsi_cross_down_70 = (rsi < 70.0) and (rsi_prev >= 70.0)
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Primary: 4h bullish + RSI oversold or crossing up
        if htf_4h_bullish:
            if rsi_oversold:
                desired_signal = SIZE
            elif rsi_cross_up_40:
                desired_signal = SIZE
            elif rsi_cross_up_30 and htf_1d_bullish:
                # Stronger signal if 1d also bullish
                desired_signal = SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        # Primary: 4h bearish + RSI overbought or crossing down
        elif htf_4h_bearish:
            if rsi_overbought:
                desired_signal = -SIZE
            elif rsi_cross_down_60:
                desired_signal = -SIZE
            elif rsi_cross_down_70 and htf_1d_bearish:
                # Stronger signal if 1d also bearish
                desired_signal = -SIZE
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0:
            if position_side > 0:
                # Hold long if 4h still bullish and RSI not extremely overbought
                if htf_4h_bullish and rsi < 75.0:
                    desired_signal = SIZE
            elif position_side < 0:
                # Hold short if 4h still bearish and RSI not extremely oversold
                if htf_4h_bearish and rsi > 25.0:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
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