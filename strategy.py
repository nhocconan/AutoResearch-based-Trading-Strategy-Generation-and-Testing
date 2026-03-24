#!/usr/bin/env python3
"""
Experiment #185: 15m Primary + 4h/1d HTF — Simplified Trend-Pullback with Session Filter

Hypothesis: 15m strategies failed because entry conditions were TOO STRICT (0 trades).
This version SIMPLIFIES entries while keeping HTF bias for direction.

Core Logic:
- 4h HMA(21) slope determines trend bias (long-only or short-only)
- 1d HMA(50) confirms major regime (avoid counter-trend in strong trends)
- 15m RSI(7) extreme for entry timing (oversold in uptrend, overbought in downtrend)
- Session filter: 00-12 UTC (London/NY overlap = higher volume, cleaner moves)
- ATR trailing stop for risk management

Why this should work:
- Fewer filters = more trades (previous 15m got 0 trades from over-filtering)
- HTF bias prevents whipsaw (don't long in 4h downtrend)
- RSI(7) is fast enough for 15m but not too noisy
- Session filter reduces low-volume chop (Asia session often fake moves)

Position sizing: 0.20 base, 0.30 strong confluence (smaller due to higher frequency)
Target: 50-80 trades/year, Sharpe > 0.40, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_trend_rsi_session_4h1d_v1"
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

def calculate_hma_slope(hma, lookback=5):
    """Calculate HMA slope over lookback period"""
    n = len(hma)
    slope = np.zeros(n)
    slope[:] = np.nan
    
    for i in range(lookback, n):
        if not np.isnan(hma[i]) and not np.isnan(hma[i-lookback]):
            slope[i] = (hma[i] - hma[i-lookback]) / hma[i-lookback] * 100.0
    
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 4h HMA slope (trend direction)
    hma_4h_slope = calculate_hma_slope(hma_4h_aligned, lookback=3)
    
    # Calculate and align 1d HMA for regime filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (15m) indicators
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    hma_15m = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20  # 20% base position size (15m = higher frequency)
    SIZE_STRONG = 0.30  # 30% for strong confluence
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_4h_slope[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(rsi_7[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC = London + NY overlap) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_good_session = 0 <= hour_utc <= 12
        
        # === 4h TREND BIAS ===
        # Positive slope = bullish, Negative slope = bearish
        hma_4h_bullish = hma_4h_slope[i] > 0.0
        hma_4h_bearish = hma_4h_slope[i] < 0.0
        
        # === 1d REGIME FILTER ===
        # Price above 1d HMA = bull market (prefer longs)
        # Price below 1d HMA = bear market (prefer shorts)
        bull_regime = close[i] > hma_1d_aligned[i]
        bear_regime = close[i] < hma_1d_aligned[i]
        
        # === 15m RSI EXTREMES ===
        rsi_oversold = rsi_7[i] < 35.0  # Less strict than 30 to get more trades
        rsi_overbought = rsi_7[i] > 65.0  # Less strict than 70 to get more trades
        
        # === 15m HMA CONFIRMATION ===
        hma_15m_bull = close[i] > hma_15m[i]
        hma_15m_bear = close[i] < hma_15m[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: 4h bullish + RSI oversold + (1d bull OR neutral) + session
        if hma_4h_bullish and rsi_oversold:
            confluence_count = 0
            if bull_regime:
                confluence_count += 1
            if hma_15m_bull:
                confluence_count += 1
            if in_good_session:
                confluence_count += 1
            
            if confluence_count >= 2:  # Need at least 2 of 3 confirmations
                if confluence_count >= 3:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT ENTRY: 4h bearish + RSI overbought + (1d bear OR neutral) + session
        elif hma_4h_bearish and rsi_overbought:
            confluence_count = 0
            if bear_regime:
                confluence_count += 1
            if hma_15m_bear:
                confluence_count += 1
            if in_good_session:
                confluence_count += 1
            
            if confluence_count >= 2:  # Need at least 2 of 3 confirmations
                if confluence_count >= 3:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
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