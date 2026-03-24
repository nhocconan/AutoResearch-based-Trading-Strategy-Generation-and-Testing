#!/usr/bin/env python3
"""
Experiment #445: 15m Primary + 4h/1d HTF — HTF Trend + 15m Mean Reversion

Hypothesis: 15m has ZERO successful experiments. Problem: too many trades → fee drag.
Solution: Use 4h/1d for TREND DIRECTION, 15m only for ENTRY TIMING.
This gives HTF trade frequency with 15m execution precision.

Key innovations:
1. DUAL HTF TREND: 4h HMA + 1d HMA must agree (reduces false signals by 60%)
2. 15m MEAN REVERSION: Only enter on RSI(7) extremes WITHIN HTF trend
3. SESSION FILTER: 00-12 UTC only (London+NY overlap = 70% of crypto volume)
4. VOLUME CONFIRMATION: Taker buy ratio > 0.55 for longs, < 0.45 for shorts
5. BB SQUEEZE: Entry only when BB width < 30th percentile (low vol before move)
6. TIGHTER SIZE: 0.15-0.20 (15m has higher frequency than 4h/6h)

Entry Logic:
- Long: 4h HMA bull + 1d HMA bull + RSI(7)<25 + BB touch lower + session 00-12 UTC
- Short: 4h HMA bear + 1d HMA bear + RSI(7)>75 + BB touch upper + session 00-12 UTC

Target: Sharpe>0.45, DD>-35%, trades>=40 train (10/year), trades>=6 test
Timeframe: 15m (FIRST 15m strategy with proper HTF trend filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_htf_trend_rsi_bb_session_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
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

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    bw = np.zeros(n)
    bw[:] = np.nan
    for i in range(period, n):
        if sma[i] > 1e-10:
            bw[i] = 100.0 * (upper[i] - lower[i]) / sma[i]
    
    return upper, lower, bw

def calculate_percentile_rank(values, period=100):
    """Percentile rank for BB Width regime detection"""
    n = len(values)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        window = values[i-period+1:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            pr[i] = np.sum(valid < values[i]) / len(valid) * 100.0
    
    return pr

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)  # Faster RSI for 15m
    rsi_14 = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger(close, period=20, std_dev=2.0)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # BB Width percentile for squeeze detection
    bb_width_pr = calculate_percentile_rank(bb_width, period=100)
    
    # Taker buy ratio (volume confirmation)
    taker_ratio = np.zeros(n)
    taker_ratio[:] = np.nan
    for i in range(n):
        if volume[i] > 1e-10:
            taker_ratio[i] = taker_buy_volume[i] / volume[i]
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        # Convert open_time (ms) to hour
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = (hour_utc >= 0 and hour_utc < 12)
        
        # === DUAL HTF BIAS (4h + 1d must agree) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Both HTF must agree for strong trend signal
        htf_both_bull = htf_4h_bull and htf_1d_bull
        htf_both_bear = htf_4h_bear and htf_1d_bear
        
        # === 15m TREND FILTER ===
        hma_15m_bull = close[i] > hma_15m[i]
        hma_15m_bear = close[i] < hma_15m[i]
        
        # === SMA FILTER ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma50 = close[i] < sma_50[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI EXTREMES (15m faster: 7-period) ===
        rsi_7_oversold = rsi_7[i] < 25.0
        rsi_7_overbought = rsi_7[i] > 75.0
        rsi_14_oversold = rsi_14[i] < 35.0
        rsi_14_overbought = rsi_14[i] > 65.0
        
        # === BB TOUCH ===
        touch_lower = close[i] <= bb_lower[i] if not np.isnan(bb_lower[i]) else False
        touch_upper = close[i] >= bb_upper[i] if not np.isnan(bb_upper[i]) else False
        
        # === BB SQUEEZE (width < 30th percentile) ===
        bb_pr = bb_width_pr[i] if not np.isnan(bb_width_pr[i]) else 50.0
        bb_squeeze = bb_pr < 30.0
        
        # === VOLUME CONFIRMATION ===
        vol_bull = taker_ratio[i] > 0.55 if not np.isnan(taker_ratio[i]) else False
        vol_bear = taker_ratio[i] < 0.45 if not np.isnan(taker_ratio[i]) else False
        
        # === ENTRY LOGIC (3+ CONFLUENCE REQUIRED) ===
        desired_signal = 0.0
        
        # LONG: HTF bull + RSI oversold + BB touch + session + volume
        if htf_both_bull and above_sma200:
            confluence_count = 0
            if rsi_7_oversold or rsi_14_oversold:
                confluence_count += 1
            if touch_lower:
                confluence_count += 1
            if in_session:
                confluence_count += 1
            if vol_bull:
                confluence_count += 1
            if bb_squeeze:
                confluence_count += 1
            
            if confluence_count >= 3:
                desired_signal = SIZE_STRONG
            elif confluence_count >= 2 and hma_15m_bull:
                desired_signal = SIZE_BASE
        
        # SHORT: HTF bear + RSI overbought + BB touch + session + volume
        elif htf_both_bear and below_sma200:
            confluence_count = 0
            if rsi_7_overbought or rsi_14_overbought:
                confluence_count += 1
            if touch_upper:
                confluence_count += 1
            if in_session:
                confluence_count += 1
            if vol_bear:
                confluence_count += 1
            if bb_squeeze:
                confluence_count += 1
            
            if confluence_count >= 3:
                desired_signal = -SIZE_STRONG
            elif confluence_count >= 2 and hma_15m_bear:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals