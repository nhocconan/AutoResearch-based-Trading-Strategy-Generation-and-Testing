#!/usr/bin/env python3
"""
Experiment #016: 4h HMA Trend + RSI Pullback with 1d/1w Regime Filter
Hypothesis: 4h captures medium-term swings while 1d HMA provides trend bias.
Add 1w HMA for macro regime (bull/bear). Asymmetric entries: easier long in bull,
easier short in bear. RSI pullback to EMA21 for entries (proven in research).
Conservative sizing (0.25-0.30) with 2.5*ATR trailing stop.
Timeframe: 4h (REQUIRED), HTF: 1d and 1w via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_1d_1w_regime_asymmetric_v1"
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

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = (close - rolling_mean) / (rolling_std + 1e-10)
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    zscore = calculate_zscore(close, 20)
    
    # HMA on 4h for trend
    hma_4h = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    max_profit = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(zscore[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (HTF) ===
        # 1w HMA = macro regime (bull/bear)
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # 1d HMA = intermediate trend
        trend_bull = close[i] > hma_1d_aligned[i]
        trend_bear = close[i] < hma_1d_aligned[i]
        
        # 4h HMA = short-term trend
        st_bull = close[i] > hma_4h[i]
        st_bear = close[i] < hma_4h[i]
        
        # EMA alignment
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === ASYMMETRIC ENTRY LOGIC ===
        # In bull regime: easier to long, harder to short
        # In bear regime: easier to short, harder to long
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: RSI pullback to 40-50 in bull regime + trend aligned
        if macro_bull and trend_bull:
            # RSI pullback entry (buy the dip)
            if rsi[i] >= 35 and rsi[i] <= 50 and st_bull and ema_bullish:
                new_signal = SIZE_ENTRY
            # Z-score oversold in bull trend
            elif zscore[i] < -1.5 and trend_bull and rsi[i] < 45:
                new_signal = SIZE_ENTRY
        elif macro_bear and trend_bull:
            # Counter-trend long in bear (only strong signals)
            if rsi[i] < 30 and zscore[i] < -2.0:
                new_signal = SIZE_ENTRY * 0.5  # Smaller size for counter-trend
        
        # === SHORT ENTRY ===
        # Primary: RSI rally to 50-65 in bear regime + trend aligned
        if macro_bear and trend_bear:
            # RSI rally entry (sell the rip)
            if rsi[i] >= 50 and rsi[i] <= 65 and st_bear and ema_bearish:
                new_signal = -SIZE_ENTRY
            # Z-score overbought in bear trend
            elif zscore[i] > 1.5 and trend_bear and rsi[i] > 55:
                new_signal = -SIZE_ENTRY
        elif macro_bull and trend_bear:
            # Counter-trend short in bull (only strong signals)
            if rsi[i] > 70 and zscore[i] > 2.0:
                new_signal = -SIZE_ENTRY * 0.5  # Smaller size for counter-trend
        
        # === STOPLOSS LOGIC ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
                max_profit = max(max_profit, close[i] - entry_price)
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
                max_profit = max(max_profit, entry_price - close[i])
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # === TAKE PROFIT (partial at 2R) ===
        if position_side != 0 and max_profit > 0:
            risk = abs(entry_price - trailing_stop)
            if risk > 0 and max_profit >= 2.0 * risk:
                # Reduce to half position at 2R profit
                if new_signal == 0.0:
                    new_signal = SIZE_HALF * np.sign(position_side)
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            max_profit = 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            max_profit = 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            max_profit = 0.0
        
        signals[i] = new_signal
    
    return signals