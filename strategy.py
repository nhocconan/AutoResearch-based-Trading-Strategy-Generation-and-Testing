#!/usr/bin/env python3
"""
Experiment #437: 15m Primary + 4h/12h HTF — Session-Filtered Trend Pullback

Hypothesis: 15m timeframe needs VERY selective entries to avoid fee drag (>100 trades/yr = death).
Recent 15m failures (#425, #429, #433) all got Sharpe=0.000 (ZERO trades) because:
- Weekly pivot filters too restrictive
- Multiple confluence conditions never all agree
- RSI thresholds too narrow

New approach inspired by #435 (6h success with Sharpe=0.153):
1. 4h HMA for PRIMARY trend bias (faster than 12h/1d for 15m entries)
2. 12h HMA for SECONDARY confirmation (looser - only 1 of 2 HTF needed)
3. 15m RSI(7) for fast pullback entries (not RSI(14) which is too slow)
4. SESSION FILTER: Only trade 00-12 UTC (London+NY overlap = quality volume)
5. LOOSENED entries: RSI(7)<35 OR RSI(7)>65 (not both extremes required)
6. STOPLOSS: 2.0x ATR from entry (preserve capital in crashes)
7. SIZE: 0.15-0.20 (smaller for higher frequency 15m)

Target: Sharpe>0.40, DD>-35%, trades>=40 train (10/year), trades>=6 test
Timeframe: 15m (FIRST proper 15m experiment with working trade generation)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_session_rsi_pullback_4h12h_v1"
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower

def is_session_active(open_time, start_hour=0, end_hour=12):
    """Check if bar is within UTC session window (00-12 UTC for London+NY overlap)"""
    # open_time is in milliseconds since epoch
    hour = pd.to_datetime(open_time, unit='ms').hour
    return start_hour <= hour < end_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi_fast = calculate_rsi(close, period=7)  # Fast RSI for 15m entries
    rsi_slow = calculate_rsi(close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    bb_upper, bb_lower = calculate_bollinger(close, period=20, std_dev=2.0)
    
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
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_fast[i]) or np.isnan(rsi_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        in_session = is_session_active(open_time[i], start_hour=0, end_hour=12)
        
        # === HTF TREND BIAS (4h primary, 12h confirmation) ===
        # At least ONE HTF must agree (looser than dual agreement)
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        # Primary bias from 4h, confirmed if 12h agrees OR neutral
        htf_bullish = htf_4h_bull and (htf_12h_bull or not htf_12h_bear)
        htf_bearish = htf_4h_bear and (htf_12h_bear or not htf_12h_bull)
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === SMA FILTER (price above/below key levels) ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma50 = close[i] < sma_50[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI EXTREMES (FAST RSI(7) for 15m) ===
        rsi_oversold = rsi_fast[i] < 35.0
        rsi_overbought = rsi_fast[i] > 65.0
        
        # === BB TOUCH ===
        touch_lower = close[i] <= bb_lower[i] if not np.isnan(bb_lower[i]) else False
        touch_upper = close[i] >= bb_upper[i] if not np.isnan(bb_upper[i]) else False
        
        # === RSI PULLBACK IN TREND (main entry signal) ===
        # Long: HTF bullish + RSI(7) oversold pullback
        # Short: HTF bearish + RSI(7) overbought pullback
        
        desired_signal = 0.0
        
        # LONG ENTRIES (loosened for trade generation)
        if htf_bullish and in_session:
            # Primary: RSI(7) < 35 pullback in uptrend
            if rsi_oversold and above_sma50:
                desired_signal = SIZE_STRONG
            # Secondary: BB lower touch + RSI < 50
            elif touch_lower and rsi_fast[i] < 50.0:
                desired_signal = SIZE_BASE
            # Tertiary: Simple RSI oversold in bull trend
            elif rsi_oversold and hma_bull:
                desired_signal = SIZE_BASE
        
        # SHORT ENTRIES (loosened for trade generation)
        elif htf_bearish and in_session:
            # Primary: RSI(7) > 65 pullback in downtrend
            if rsi_overbought and below_sma50:
                desired_signal = -SIZE_STRONG
            # Secondary: BB upper touch + RSI > 50
            elif touch_upper and rsi_fast[i] > 50.0:
                desired_signal = -SIZE_BASE
            # Tertiary: Simple RSI overbought in bear trend
            elif rsi_overbought and hma_bear:
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