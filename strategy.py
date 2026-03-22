#!/usr/bin/env python3
"""
Experiment #003: 1h RSI Mean Reversion with 4h Trend Bias - Simplified Entry Logic
Hypothesis: Previous strategy had too many conflicting filters causing 0 trades.
This version uses SIMPLER entry conditions: RSI extremes (<30 long, >70 short) 
aligned with 4h HMA trend bias. Fewer filters = more trades while maintaining edge.
Key changes from failed #001:
1. Reduced RSI thresholds (30/70 instead of 25/75) for more triggers
2. Removed regime detection (BB Width percentile was too restrictive)
3. Simplified position tracking - no complex take-profit reduction
4. Single entry path per direction with HTF confirmation
5. Discrete sizing (0.30) with 2.5*ATR stoploss
Must generate >=10 trades per symbol on train, >=3 on test.
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_mean_revert_4h_hma_simple_v2"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_sma(close, period):
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
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
    # Additional trend filter
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position size (30% of capital)
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    stop_price = 0.0
    
    for i in range(200, n):  # Start after SMA200 is ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - primary filter
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # RSI signals - relaxed thresholds for more trades
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Additional confirmation - price above/below SMA200
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # EMA alignment for trend confirmation
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Path 1: 4h bullish + RSI oversold + above SMA200 (trend pullback)
        if hma_4h_bullish and rsi_oversold and above_sma200:
            new_signal = SIZE
        # Path 2: 4h bullish + RSI oversold + EMA bullish (stronger confirmation)
        elif hma_4h_bullish and rsi_oversold and ema_bullish:
            new_signal = SIZE
        # Path 3: RSI very oversold (<20) + above SMA200 (any 4h trend)
        elif rsi[i] < 20 and above_sma200:
            new_signal = SIZE
        
        # === SHORT ENTRY ===
        # Path 1: 4h bearish + RSI overbought + below SMA200 (trend pullback)
        if hma_4h_bearish and rsi_overbought and below_sma200:
            new_signal = -SIZE
        # Path 2: 4h bearish + RSI overbought + EMA bearish (stronger confirmation)
        elif hma_4h_bearish and rsi_overbought and ema_bearish:
            new_signal = -SIZE
        # Path 3: RSI very overbought (>80) + below SMA200 (any 4h trend)
        elif rsi[i] > 80 and below_sma200:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update stop price (2.5*ATR trailing)
            current_stop = entry_price - 2.5 * atr[i]
            if close[i] < current_stop:
                new_signal = 0.0  # Stoploss hit
        
        if position_side < 0 and entry_price > 0:
            # Update stop price (2.5*ATR trailing)
            current_stop = entry_price + 2.5 * atr[i]
            if close[i] > current_stop:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
        
        signals[i] = new_signal
    
    return signals