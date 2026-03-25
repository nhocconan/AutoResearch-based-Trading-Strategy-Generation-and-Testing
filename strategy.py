#!/usr/bin/env python3
"""
Experiment #1413: 5m Primary + 15m/4h HTF — Session-Filtered Trend Pullback

Hypothesis: 5m timeframe is unexplored territory. Key insights:
1. 5m needs EXTREME selectivity (50-120 trades/year max) or fee drag kills returns
2. MUST use 4h/15m HTF for trend direction - never trade counter-trend on 5m
3. Session filter (08-20 UTC) captures London/NY overlap = highest volume/liquidity
4. 5m RSI pullback into 4h trend = proven pattern with tight entries
5. Small position size (0.15-0.20) due to higher trade frequency

Why this might work where 15m failed:
- 5m entries are tighter (less slippage on entry)
- Session filter eliminates Asian session chop (00-08 UTC)
- 4h + 15m dual HTF filter = stronger trend confirmation than single HTF
- Discrete sizing (0.15/0.20) minimizes fee churn on signal changes

Entry logic (LOOSE enough to generate trades):
- LONG: 4h_HMA bullish + 15m_HMA bullish + 5m_RSI 35-55 (pullback) + session 08-20 UTC
- SHORT: 4h_HMA bearish + 15m_HMA bearish + 5m_RSI 45-65 (pullback) + session 08-20 UTC

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 5m
Size: 0.15-0.20 discrete (smaller due to more trades)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_hma_rsi_pullback_15m4h_v1"
timeframe = "5m"
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

def calculate_momentum(close, period=10):
    """Rate of Change momentum"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    mom = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] != 0:
            mom[i] = (close[i] - close[i - period]) / close[i - period] * 100
    return mom

def is_session_active(open_time_unix):
    """
    Check if bar is within active session (08-20 UTC)
    open_time_unix: Unix timestamp in milliseconds
    Returns True if hour is between 08-20 UTC
    """
    # Convert ms to seconds, then to datetime
    ts_seconds = open_time_unix / 1000.0
    hour = pd.to_datetime(ts_seconds, unit='s').hour
    return 8 <= hour < 20

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_15m_21_raw = calculate_hma(df_15m['close'].values, period=21)
    hma_15m_21_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_21_raw)
    
    hma_4h_21_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21_raw)
    
    # Calculate 5m indicators
    hma_5m_16 = calculate_hma(close, period=16)
    hma_5m_48 = calculate_hma(close, period=48)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    mom_10 = calculate_momentum(close, period=10)
    
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
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_5m_16[i]) or np.isnan(hma_5m_48[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m_21_aligned[i]) or np.isnan(hma_4h_21_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC only) ===
        session_active = is_session_active(open_time[i])
        
        if not session_active:
            # Outside session - flatten position
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        # === TREND DIRECTION (4h HMA bias - strongest) ===
        price_above_4h = close[i] > hma_4h_21_aligned[i]
        price_below_4h = close[i] < hma_4h_21_aligned[i]
        
        # === INTERMEDIATE TREND (15m HMA confirmation) ===
        price_above_15m = close[i] > hma_15m_21_aligned[i]
        price_below_15m = close[i] < hma_15m_21_aligned[i]
        
        # === 5m HMA CROSSOVER (short-term momentum) ===
        hma_5m_bullish = hma_5m_16[i] > hma_5m_48[i]
        hma_5m_bearish = hma_5m_16[i] < hma_5m_48[i]
        
        # === RSI PULLBACK (LOOSE entry - guarantee trades) ===
        rsi = rsi_14[i]
        mom = mom_10[i] if not np.isnan(mom_10[i]) else 0.0
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + 15m bullish + 5m RSI pullback (35-55) + momentum positive
        if price_above_4h and price_above_15m and rsi > 35 and rsi < 55:
            if mom > -0.5:  # Slight momentum confirmation
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bearish + 15m bearish + 5m RSI pullback (45-65) + momentum negative
        elif price_below_4h and price_below_15m and rsi > 45 and rsi < 65:
            if mom < 0.5:  # Slight momentum confirmation
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
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
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
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