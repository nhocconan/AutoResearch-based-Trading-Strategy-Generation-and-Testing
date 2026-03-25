#!/usr/bin/env python3
"""
Experiment #1441: 15m Primary + 1h/4h HTF — Triple Timeframe Trend Pullback

Hypothesis: 15m has ZERO successful experiments because entry conditions were TOO STRICT.
This strategy uses LOOSE filters to guarantee trades while maintaining quality:

1. 4h HMA(21) for major trend bias (price vs HMA, simple and effective)
2. 1h RSI(14) for momentum (RSI > 45 long, < 55 short - very loose)
3. 15m EMA(8) vs EMA(21) for entry timing (crossover signals)
4. Session filter: 00-12 UTC only (London+NY overlap, reduces trade count)
5. ATR(14) trailing stoploss at 2.5x (signal→0 when stopped)
6. Discrete sizing: 0.0, ±0.15, ±0.20 (15m needs smaller size due to frequency)

Why this should work where other 15m strategies failed:
- LOOSE RSI thresholds (45/55 vs 30/70) = more trades
- Only 3 filters max (HTF trend + momentum + entry) = less conflict
- Session filter controls trade frequency (target 50-80 trades/year)
- Smaller position size (0.15-0.20) = lower drawdown on 15m volatility
- Simple EMA crossover on 15m = frequent but HTF-filtered signals

Entry logic (LOOSE to guarantee trades):
- LONG: 4h_price > 4h_HMA + 1h_RSI > 45 + 15m_EMA8 > EMA21 + UTC 00-12
- SHORT: 4h_price < 4h_HMA + 1h_RSI < 55 + 15m_EMA8 < EMA21 + UTC 00-12

Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for 15m frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_triple_tf_hma_rsi_ema_session_v1"
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

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate 15m indicators
    ema_8 = calculate_ema(close, period=8)
    ema_21 = calculate_ema(close, period=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
        
        if np.isnan(ema_8[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = 0 <= hour_utc <= 12
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === MOMENTUM (1h RSI - LOOSE thresholds) ===
        rsi_1h = rsi_1h_aligned[i]
        rsi_bullish = rsi_1h > 45  # Very loose (not 50)
        rsi_bearish = rsi_1h < 55  # Very loose (not 50)
        
        # === ENTRY TIMING (15m EMA crossover) ===
        ema_bullish = ema_8[i] > ema_21[i]
        ema_bearish = ema_8[i] < ema_21[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        if in_session:
            # LONG: 4h bullish + 1h RSI > 45 + 15m EMA bullish
            if price_above_4h and rsi_bullish and ema_bullish:
                desired_signal = SIZE_STRONG
            
            # SHORT: 4h bearish + 1h RSI < 55 + 15m EMA bearish
            elif price_below_4h and rsi_bearish and ema_bearish:
                desired_signal = -SIZE_STRONG
        
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