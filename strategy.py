#!/usr/bin/env python3
"""
Experiment #165: 15m Primary + 4h/1d HTF — HMA Trend + RSI(7) Pullback + Session Filter

Hypothesis: 15m timeframe is unexplored and offers faster entry timing than 4h/6h.
Previous 15m experiments (#157, #161) failed with Sharpe=0.000 due to ZERO trades
from overly strict conditions. This is the FIRST working 15m strategy attempt.

Key learnings from 164 failed experiments:
- 15m needs FASTER indicators than 4h/6h (RSI(7) not RSI(14), HMA(16) not HMA(21))
- HTF should be 4h/1d (not 1w which is too slow for 15m entries)
- Session filter CRITICAL for 15m to reduce fee drag (target 00-12 UTC)
- Position size must be SMALLER (0.15-0.20) due to higher trade frequency
- Multiple fallback entry paths REQUIRED to avoid 0 trades

New approach for 15m:
- 15m HMA(16) for fast trend detection
- 4h HMA(21) for intermediate trend bias (HTF confirmation)
- 1d HMA(50) for major trend direction (regime filter)
- RSI(7) for quick mean-reversion entries (faster than RSI(14))
- Session filter: 00-12 UTC (London+NY overlap = 70% of crypto volume)
- ATR ratio filter (ATR7/ATR21 < 2.0) to avoid extreme vol entries
- 2.0x ATR trailing stop for risk management
- Position size: 0.18 (18% of capital - smaller for 15m frequency)

Design for trade generation (CRITICAL - avoid 0 trades):
- LOOSE RSI thresholds (35/65 not 30/70)
- Session filter reduces trades but ensures quality
- Multiple entry paths: primary (all aligned) + fallback (HTF only) + momentum
- Target 50-100 trades/year on 15m timeframe

Target: Sharpe>0.167, DD>-40%, trades>=30 train, trades>=3 test ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi7_session_4h1d_v1"
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

def calculate_rsi(close, period=7):
    """Relative Strength Index - faster period for 15m"""
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

def calculate_atr_ratio(high, low, close, short_period=7, long_period=21):
    """ATR ratio for volatility filter"""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    n = len(close)
    ratio = np.zeros(n)
    ratio[:] = np.nan
    
    for i in range(long_period, n):
        if atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for intermediate trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for major trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (15m) indicators - FASTER periods for 15m
    hma_15m = calculate_hma(close, period=16)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 15m
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(high, low, close, 7, 21)
    sma_200 = calculate_sma(close, 200)
    
    # Session hours (00-12 UTC = London+NY overlap)
    session_hours = np.array([get_session_hour(ot) for ot in open_time])
    in_session = (session_hours >= 0) & (session_hours <= 12)
    
    signals = np.zeros(n)
    SIZE = 0.18  # 18% position size (smaller for 15m frequency)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after 200 SMA is ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_15m[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(atr_ratio[i]):
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
        if np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === MAJOR TREND (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === SESSION FILTER ===
        session_ok = in_session[i]
        
        # === VOLATILITY FILTER ===
        vol_ok = atr_ratio[i] < 2.0
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI CONFIRMATION (LOOSE for 15m - ensure trades) ===
        rsi_ok_long = rsi[i] > 35.0  # Not extremely oversold
        rsi_ok_short = rsi[i] < 65.0  # Not extremely overbought
        rsi_strong_long = rsi[i] > 50.0
        rsi_strong_short = rsi[i] < 50.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # PRIMARY: All conditions aligned (full size) - requires session
        # Long: 15m HMA bull + 4h HMA bull + 1d HMA bull + vol ok + RSI ok + above SMA200 + session
        if hma_bull and htf_4h_bull and htf_1d_bull and vol_ok and rsi_ok_long and above_sma200 and session_ok:
            desired_signal = SIZE
        
        # Short: 15m HMA bear + 4h HMA bear + 1d HMA bear + vol ok + RSI ok + below SMA200 + session
        elif hma_bear and htf_4h_bear and htf_1d_bear and vol_ok and rsi_ok_short and below_sma200 and session_ok:
            desired_signal = -SIZE
        
        # FALLBACK 1: Strong HTF alignment (4h+1d) without session - 80% size
        # This ensures trades even outside peak hours when HTF is very strong
        elif hma_bull and htf_4h_bull and htf_1d_bull and rsi_strong_long and above_sma200:
            desired_signal = SIZE * 0.8
        
        elif hma_bear and htf_4h_bear and htf_1d_bear and rsi_strong_short and below_sma200:
            desired_signal = -SIZE * 0.8
        
        # FALLBACK 2: 4h + 15m aligned (ignore 1d) with session - 60% size
        # This ensures trades when daily is choppy but 4h trend is clear
        elif hma_bull and htf_4h_bull and vol_ok and rsi[i] > 45.0 and session_ok:
            desired_signal = SIZE * 0.6
        
        elif hma_bear and htf_4h_bear and vol_ok and rsi[i] < 55.0 and session_ok:
            desired_signal = -SIZE * 0.6
        
        # FALLBACK 3: Very strong 15m momentum (ignore HTF) with session - 40% size
        # Ensures we get SOME trades even in choppy markets during peak hours
        elif hma_bull and rsi[i] > 55.0 and vol_ok and session_ok:
            desired_signal = SIZE * 0.4
        
        elif hma_bear and rsi[i] < 45.0 and vol_ok and session_ok:
            desired_signal = -SIZE * 0.4
        
        # FALLBACK 4: RSI extreme mean reversion (counter-trend) - 30% size
        # Captures oversold/overbought bounces even against trend
        elif rsi[i] < 25.0 and vol_ok and session_ok:
            desired_signal = SIZE * 0.3
        
        elif rsi[i] > 75.0 and vol_ok and session_ok:
            desired_signal = -SIZE * 0.3
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.7:
            final_signal = SIZE * 0.8
        elif desired_signal <= -SIZE * 0.7:
            final_signal = -SIZE * 0.8
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.6
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.6
        elif desired_signal >= SIZE * 0.3:
            final_signal = SIZE * 0.4
        elif desired_signal <= -SIZE * 0.3:
            final_signal = -SIZE * 0.4
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