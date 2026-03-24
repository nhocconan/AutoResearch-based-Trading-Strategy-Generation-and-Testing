#!/usr/bin/env python3
"""
Experiment #133: 5m Primary + 15m/4h HTF — Multi-TF Pullback Strategy with Session Filter

Hypothesis: 5m timeframe has ZERO prior experiments. Key insights:
- 5m needs EXTREME selectivity to avoid fee drag (target 50-120 trades/year)
- Use 4h HMA for major trend bias (never trade counter-trend)
- Use 15m HMA for intermediate trend confirmation
- Use 5m RSI(7) for pullback entries in trend direction (buy dips in uptrend)
- Session filter MANDATORY: 09-17 UTC (London/NY overlap = highest volume)
- Small position size (0.18) due to higher trade frequency
- ATR(14) 2.5x trailing stoploss

This is a TREND-FOLLOWING PULLBACK strategy, NOT counter-trend.
Only enter when all 3 timeframes align + session filter + RSI pullback.

Key design choices:
- Timeframe: 5m (primary execution)
- HTF: 4h HMA(21) + 15m HMA(21) for trend alignment
- Entry: RSI(7) pullback to 30-40 (long) or 60-70 (short) in trend direction
- Session: 09-17 UTC only (avoid low-volume Asia night)
- Size: 0.18 (18% of capital, conservative for 5m)
- Stoploss: 2.5x ATR trailing

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 train, trades>=3 test ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_rsi_pullback_hma_15m4h_session_v1"
timeframe = "5m"
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
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_session_mask(prices):
    """
    Session filter: 09-17 UTC (London/NY overlap)
    Returns boolean mask for valid trading hours
    """
    # Parse open_time to get hour
    if 'open_time' in prices.columns:
        # open_time is in milliseconds
        hours = (prices['open_time'].values // (1000 * 60 * 60)) % 24
    else:
        # Fallback: assume index is datetime
        hours = prices.index.hour.values
    
    # Session: 09-17 UTC (inclusive start, exclusive end)
    session_mask = (hours >= 9) & (hours < 17)
    return session_mask

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF HMA for trend bias
    hma_15m_raw = calculate_hma(df_15m['close'].values, period=21)
    hma_15m_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (5m) indicators
    hma_5m = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=7)  # Fast RSI for pullback detection
    atr = calculate_atr(high, low, close, period=14)
    
    # Session filter
    session_mask = calculate_session_mask(prices)
    
    signals = np.zeros(n)
    SIZE = 0.18  # 18% position size (smaller for 5m due to more trades)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_5m[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_15m_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER ===
        in_session = session_mask[i]
        
        # === HTF TREND BIAS (4h + 15m alignment) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_15m_bull = close[i] > hma_15m_aligned[i]
        htf_15m_bear = close[i] < hma_15m_aligned[i]
        
        # === 5m HMA TREND ===
        hma_5m_bull = close[i] > hma_5m[i]
        hma_5m_bear = close[i] < hma_5m[i]
        
        # === RSI PULLBACK DETECTION ===
        # Long: RSI pulled back to 30-45 zone (oversold in uptrend)
        rsi_pullback_long = (rsi[i] >= 30.0) & (rsi[i] <= 45.0)
        # Short: RSI pulled back to 55-70 zone (overbought in downtrend)
        rsi_pullback_short = (rsi[i] >= 55.0) & (rsi[i] <= 70.0)
        
        # === ENTRY LOGIC (ALL CONDITIONS MUST ALIGN) ===
        desired_signal = 0.0
        
        # LONG: All HTF bull + 5m HMA bull + RSI pullback + in session
        if htf_4h_bull and htf_15m_bull and hma_5m_bull and rsi_pullback_long and in_session:
            desired_signal = SIZE
        
        # SHORT: All HTF bear + 5m HMA bear + RSI pullback + in session
        elif htf_4h_bear and htf_15m_bear and hma_5m_bear and rsi_pullback_short and in_session:
            desired_signal = -SIZE
        
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
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