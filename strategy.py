#!/usr/bin/env python3
"""
Experiment #1293: 5m Primary + 15m/4h HTF — Trend-Following Pullback Strategy

Hypothesis: 5m timeframe has never been tested. This strategy uses:
1. 15m HMA(21) for primary trend direction (faster than 4h, slower than 5m noise)
2. 4h HMA(21) for major regime bias (only trade with 4h trend)
3. 5m RSI(14) pullback entries (buy dips in uptrend, sell rallies in downtrend)
4. Session filter: 08:00-20:00 UTC only (London/NY overlap = highest liquidity)
5. ATR(14) 2.5x trailing stop for risk management

Why this should work on 5m:
- 5m is too noisy for pure trend following (whipsaw city)
- 5m is too noisy for pure mean reversion (trend can run far)
- SOLUTION: Mean reversion WITHIN established trend (buy dips, sell rallies)
- Session filter avoids Asian session chop (00:00-08:00 UTC)
- HTF trend filter (15m+4h) ensures we only trade with momentum
- RSI pullback = enter when trend temporarily exhausts (better entry price)

Entry logic (LOOSE to guarantee trades):
- LONG: 15m_HMA rising + 4h_HMA bullish + 5m_RSI(14) < 45 (pullback)
- SHORT: 15m_HMA falling + 4h_HMA bearish + 5m_RSI(14) > 55 (pullback)

Target: Sharpe>0.5, trades>=50 train, trades>=5 test, DD>-35%
Timeframe: 5m
Size: 0.15-0.20 discrete (smaller due to more trades = fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_trend_pullback_rsi_hma_15m4h_session_v1"
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def is_session_active(open_time_ms):
    """Check if bar is within London/NY overlap session (08:00-20:00 UTC)"""
    # open_time is in milliseconds since epoch
    dt = pd.to_datetime(open_time_ms, unit='ms', utc=True)
    hour = dt.hour
    # Active: 08:00 to 20:00 UTC (12 hours of high liquidity)
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
    hma_15m_raw = calculate_hma(df_15m['close'].values, period=21)
    hma_15m_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 5m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Also calculate 5m HMA for local trend confirmation
    hma_5m = calculate_hma(close, period=21)
    
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
        
        if np.isnan(rsi_14[i]):
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
        
        if np.isnan(hma_5m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08:00-20:00 UTC only) ===
        if not is_session_active(open_time[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (15m HMA slope + 4h HMA bias) ===
        # 15m HMA slope (compare to 3 bars ago for stability)
        hma_15m_slope = 0.0
        if i >= 3 and not np.isnan(hma_15m_aligned[i-3]):
            hma_15m_slope = hma_15m_aligned[i] - hma_15m_aligned[i-3]
        
        # 4h HMA bias (major regime)
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # 5m price vs 5m HMA for local confirmation
        price_above_5m = close[i] > hma_5m[i]
        price_below_5m = close[i] < hma_5m[i]
        
        # === MOMENTUM (RSI pullback) ===
        rsi = rsi_14[i]
        
        # === ENTRY LOGIC (Mean reversion WITHIN trend) ===
        desired_signal = 0.0
        
        # LONG: 15m HMA rising + 4h bullish + RSI oversold (pullback entry)
        if hma_15m_slope > 0 and price_above_4h:
            if rsi < 45.0:  # Pullback in uptrend
                if rsi < 35.0:
                    desired_signal = SIZE_STRONG  # Deep pullback
                else:
                    desired_signal = SIZE_BASE  # Normal pullback
        
        # SHORT: 15m HMA falling + 4h bearish + RSI overbought (pullback entry)
        elif hma_15m_slope < 0 and price_below_4h:
            if rsi > 55.0:  # Rally in downtrend
                if rsi > 65.0:
                    desired_signal = -SIZE_STRONG  # Strong rally
                else:
                    desired_signal = -SIZE_BASE  # Normal rally
        
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