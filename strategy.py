#!/usr/bin/env python3
"""
Experiment #346: 4h Bollinger Squeeze Breakout with 1d HMA Trend Filter + MACD Momentum

Hypothesis: After 294 failed strategies, the key insight is that 4h timeframe needs:
1. 1d HMA for stable major trend bias (prevents counter-trend trades in crashes)
2. Bollinger Band squeeze detection (low vol precedes big moves)
3. MACD histogram for momentum confirmation (avoids false breakouts)
4. Asymmetric logic: more aggressive longs in bull, selective shorts in bear

Why 4h works better than 1h/15m:
- Fewer false signals, less fee churn
- Captures multi-day trends without noise
- BB squeeze on 4h = meaningful volatility compression

Entry Logic:
- LONG: price > 1d HMA + BB squeeze (BW < 20th percentile) + MACD hist > 0 + RSI > 50
- SHORT: price < 1d HMA + BB squeeze + MACD hist < 0 + RSI < 50
- Exit: BB expands (squeeze ends) OR 2.5x ATR stoploss OR 1d trend flips

Position sizing: 0.28 discrete (balanced between fee churn and exposure)
Stoploss: 2.5 * ATR(14) trailing
Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_bb_squeeze_1d_hma_macd_momentum_atr_v1"
timeframe = "4h"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma * 100
    return upper.values, lower.values, bandwidth.values

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_percentile_rank(values, lookback=100):
    """Calculate percentile rank of current value vs lookback window."""
    n = len(values)
    prank = np.full(n, np.nan)
    for i in range(lookback, n):
        window = values[i-lookback:i]
        current = values[i]
        if len(window) > 0:
            prank[i] = np.sum(window < current) / len(window) * 100
    return prank

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    rsi = calculate_rsi(close, 14)
    
    # Calculate BB bandwidth percentile for squeeze detection
    bb_prank = calculate_percentile_rank(bb_bandwidth, lookback=100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_bandwidth[i]) or np.isnan(bb_prank[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(macd_hist[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === 1d HMA TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === BOLLINGER SQUEEZE DETECTION ===
        # BB bandwidth in bottom 20th percentile = squeeze (low vol before breakout)
        bb_squeeze = bb_prank[i] < 25.0
        
        # === MACD MOMENTUM ===
        macd_bullish = macd_hist[i] > 0
        macd_bearish = macd_hist[i] < 0
        
        # === RSI CONFIRMATION ===
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG: 1d bullish + BB squeeze + MACD bullish + RSI bullish
        if bull_trend_1d and bb_squeeze and macd_bullish and rsi_bullish:
            new_signal = SIZE
        
        # SHORT: 1d bearish + BB squeeze + MACD bearish + RSI bearish
        elif bear_trend_1d and bb_squeeze and macd_bearish and rsi_bearish:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === SQUEEZE RELEASE EXIT ===
        # Exit when BB bandwidth expands (squeeze ends, move likely complete)
        if in_position and new_signal != 0.0:
            bb_expanding = bb_prank[i] > 50.0  # bandwidth now above median
            if bb_expanding:
                new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 1d trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals