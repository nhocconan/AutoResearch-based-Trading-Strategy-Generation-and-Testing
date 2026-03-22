#!/usr/bin/env python3
"""
Experiment #013: 15m Mean Reversion with 4h HMA Trend Filter
Hypothesis: 15m timeframe captures short-term mean reversion opportunities while
4h HMA provides trend bias to avoid counter-trend trades. In bear/range markets
(2022, 2025), mean reversion outperforms trend following. Entry on RSI extremes
(30/70) aligned with 4h trend direction. Bollinger Band width filters low-vol
periods. ATR trailing stop at 2.5x for risk management.
Timeframe: 15m (REQUIRED), HTF: 4h via mtf_data helper.
Position sizing: 0.25-0.30 discrete levels to minimize fee churn.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_mean_revert_4h_hma_bb_rsi_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def calculate_bb_width(upper, lower, sma):
    """Calculate Bollinger Band Width as volatility measure."""
    width = (upper - lower) / sma
    return width

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

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
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_sma)
    zscore = calculate_zscore(close, 20)
    
    # Additional trend filter
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # BB width percentile for regime detection (volatility filter)
    bb_width_sma = pd.Series(bb_width).rolling(window=100, min_periods=100).mean().values
    
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
    profit_target_hit = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(bb_upper[i]) or np.isnan(zscore[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - use previous completed 4h bar
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # EMA trend confirmation on 15m
        ema_bullish = close[i] > ema_21[i] and ema_21[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and ema_21[i] < ema_50[i]
        
        # Volatility regime - avoid trading in extremely low vol
        vol_normal = bb_width[i] > 0.5 * bb_width_sma[i] if not np.isnan(bb_width_sma[i]) else True
        
        # RSI mean reversion signals
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Z-score mean reversion signals
        zscore_oversold = zscore[i] < -1.5
        zscore_overbought = zscore[i] > 1.5
        
        # Bollinger Band touch signals
        bb_touch_lower = close[i] <= bb_lower[i] * 1.001
        bb_touch_upper = close[i] >= bb_upper[i] * 0.999
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Mean reversion: RSI oversold + 4h HMA bullish (trend-aligned pullback)
        if rsi_oversold and hma_4h_bullish and vol_normal:
            new_signal = SIZE_ENTRY
        # Mean reversion: BB lower touch + Z-score oversold + 4h bullish
        elif bb_touch_lower and zscore_oversold and hma_4h_bullish:
            new_signal = SIZE_ENTRY
        # EMA bullish pullback to EMA21 + 4h bullish
        elif ema_bullish and close[i] < ema_21[i] * 1.005 and close[i] > ema_21[i] * 0.995 and hma_4h_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Mean reversion: RSI overbought + 4h HMA bearish (trend-aligned pullback)
        if rsi_overbought and hma_4h_bearish and vol_normal:
            new_signal = -SIZE_ENTRY
        # Mean reversion: BB upper touch + Z-score overbought + 4h bearish
        elif bb_touch_upper and zscore_overbought and hma_4h_bearish:
            new_signal = -SIZE_ENTRY
        # EMA bearish pullback to EMA21 + 4h bearish
        elif ema_bearish and close[i] > ema_21[i] * 0.995 and close[i] < ema_21[i] * 1.005 and hma_4h_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
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
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # === TAKE PROFIT LOGIC ===
        # Reduce position at 2R profit
        if position_side > 0 and entry_price > 0:
            profit = (close[i] - entry_price) / atr[i]
            if profit >= 2.0 and not profit_target_hit:
                new_signal = SIZE_HALF  # Take half profit
                profit_target_hit = True
            elif profit >= 3.0:
                new_signal = 0.0  # Full exit at 3R
        
        if position_side < 0 and entry_price > 0:
            profit = (entry_price - close[i]) / atr[i]
            if profit >= 2.0 and not profit_target_hit:
                new_signal = -SIZE_HALF  # Take half profit
                profit_target_hit = True
            elif profit >= 3.0:
                new_signal = 0.0  # Full exit at 3R
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            profit_target_hit = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            profit_target_hit = False
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            profit_target_hit = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal) and np.sign(new_signal) == np.sign(prev_signal):
            profit_target_hit = True
        
        signals[i] = new_signal
    
    return signals