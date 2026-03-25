#!/usr/bin/env python3
"""
Experiment #1521: 15m Primary + 1h/4h/1d HTF — Camarilla Pivot Mean Reversion

Hypothesis: 15m timeframe with STRICT multi-timeframe confluence can capture
intraday mean-reversion while respecting higher-timeframe trend direction.
Key innovation: Camarilla pivot levels (R3/S3 for mean-reversion, R4/S4 for breakout)
combined with 4h trend bias and session filtering.

Why 15m can work (unlike previous failed attempts):
1. 4h HMA(21) for major trend bias — ONLY trade in HTF trend direction
2. 1h RSI(7) for momentum confirmation — avoid entering against momentum
3. Camarilla pivots from 1d HTF — institutional support/resistance levels
4. Session filter (00-12 UTC) — trade only during high liquidity
5. Very strict entry: need ALL 4 confluences (HTF trend + momentum + level + session)
6. Conservative sizing: 0.15-0.20 (smaller due to higher frequency)

Entry logic (MUST generate ≥10 trades/train, ≥3/test):
- LONG: 4h_HMA bullish + 1h_RSI < 40 (oversold) + price < S3 + session active
- SHORT: 4h_HMA bearish + 1h_RSI > 60 (overbought) + price > R3 + session active
- BREAKOUT LONG: 4h_HMA bullish + price > R4 + volume spike
- BREAKOUT SHORT: 4h_HMA bearish + price < S4 + volume spike

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%, trades/year < 100
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_camarilla_pivot_4h1h1d_session_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_camarilla_pivots(high, low, close, prev_close):
    """
    Camarilla Pivot Points - institutional support/resistance levels
    R4/S4 = breakout levels, R3/S3 = mean-reversion levels
    """
    n = len(close)
    pivot_range = prev_close - np.roll(prev_close, 1)
    pivot_range[0] = 0
    
    # Camarilla calculations
    hlc3 = (high + low + close) / 3.0
    
    r4 = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    s4 = np.full(n, np.nan)
    pivot = np.full(n, np.nan)
    
    for i in range(1, n):
        range_val = prev_close[i-1]  # Use previous day's close as reference
        if i > 1:
            high_prev = high[i-1]
            low_prev = low[i-1]
            close_prev = close[i-1]
            pivot_range_val = high_prev - low_prev
            
            pivot[i] = (high_prev + low_prev + close_prev) / 3.0
            r4[i] = close_prev + (pivot_range_val * 1.5)
            r3[i] = close_prev + (pivot_range_val * 1.25)
            s3[i] = close_prev - (pivot_range_val * 1.25)
            s4[i] = close_prev - (pivot_range_val * 1.5)
    
    return r4, r3, s3, s4, pivot

def calculate_volume_spike(volume, period=20):
    """Detect volume spikes (>2x average)"""
    n = len(volume)
    if n < period:
        return np.full(n, False)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    spike = volume > (vol_avg * 2.0)
    return spike

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=7)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Get 1d previous close for Camarilla pivots
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    r4_1d, r3_1d, s3_1d, s4_1d, pivot_1d = calculate_camarilla_pivots(
        high_1d, low_1d, close_1d, close_1d
    )
    
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    vol_spike = calculate_volume_spike(volume, period=20)
    
    # Session hours (UTC): 00-12 for London/NY overlap liquidity
    session_hours = np.array([get_session_hour(ot) for ot in open_time])
    session_active = (session_hours >= 0) & (session_hours <= 12)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_1h_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 4h TREND BIAS (major direction) ===
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # === 1h MOMENTUM (RSI confirmation) ===
        rsi_1h = rsi_1h_aligned[i]
        momentum_oversold = rsi_1h < 40
        momentum_overbought = rsi_1h > 60
        
        # === CAMARILLA LEVELS ===
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        
        # === 15m RSI (entry timing) ===
        rsi_15m = rsi_14[i]
        
        # === SESSION FILTER ===
        in_session = session_active[i]
        
        # === ENTRY LOGIC (STRICT - 4 confluences required) ===
        desired_signal = 0.0
        
        # MEAN REVERSION LONG: 4h bullish + 1h oversold + price < S3 + session
        if trend_bullish and momentum_oversold and in_session:
            if not np.isnan(s3) and close[i] < s3 * 1.002:  # within 0.2% of S3
                desired_signal = SIZE_BASE
        
        # MEAN REVERSION SHORT: 4h bearish + 1h overbought + price > R3 + session
        if trend_bearish and momentum_overbought and in_session:
            if not np.isnan(r3) and close[i] > r3 * 0.998:  # within 0.2% of R3
                desired_signal = -SIZE_BASE
        
        # BREAKOUT LONG: 4h bullish + price > R4 + volume spike
        if trend_bullish and vol_spike[i]:
            if not np.isnan(r4) and close[i] > r4:
                desired_signal = SIZE_STRONG
        
        # BREAKOUT SHORT: 4h bearish + price < S4 + volume spike
        if trend_bearish and vol_spike[i]:
            if not np.isnan(s4) and close[i] < s4:
                desired_signal = -SIZE_STRONG
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
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
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals