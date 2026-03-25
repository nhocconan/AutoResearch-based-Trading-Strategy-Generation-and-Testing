#!/usr/bin/env python3
"""
Experiment #1497: 15m Primary + 4h/12h HTF — Loose Pullback Strategy

Hypothesis: 15m timeframe with VERY LOOSE entry conditions will generate sufficient
trades while 4h/12h HTF filters prevent major counter-trend disasters.

Key insight from failures: Strategies with Sharpe=0.000 have ZERO trades.
Entry conditions were TOO STRICT (RSI<30, multiple confluence requirements).

This strategy uses:
1. 4h HMA(21) for major trend bias (aligned properly via mtf_data)
2. 12h HMA(21) for secondary confirmation
3. 15m RSI(7) with LOOSE thresholds (40/60 instead of 30/70)
4. 15m HMA(8/21) for momentum
5. ATR(14) stoploss at 2.0x
6. Discrete sizing: 0.0, ±0.15, ±0.20 (smaller for 15m frequency)

CRITICAL FOR TRADES:
- RSI long trigger: <50 (not <30) when 4h trend bullish
- RSI short trigger: >50 (not >70) when 4h trend bearish
- Only 2 confluence required (HTF trend + RSI), not 3+
- This guarantees 50-100 trades/year target

Timeframe: 15m
Size: 0.15-0.20 (smaller due to higher frequency)
Target: Sharpe>0.5, trades>=40/train, trades>=5/test, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_loose_pullback_hma_rsi_4h12h_v1"
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

def calculate_rsi(close, period=7):
    """Relative Strength Index - shorter period for 15m"""
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
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 15m indicators
    hma_8 = calculate_hma(close, period=8)
    hma_21 = calculate_hma(close, period=21)
    sma_50 = calculate_sma(close, period=50)
    sma_200 = calculate_sma(close, period=200)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    
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
    min_bars = 250
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(hma_8[i]) or np.isnan(hma_21[i]):
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
        
        if np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # 4h HMA slope (trend strength)
        hma_4h_slope_bullish = hma_4h_aligned[i] > hma_4h_aligned[i-1] if not np.isnan(hma_4h_aligned[i-1]) else False
        hma_4h_slope_bearish = hma_4h_aligned[i] < hma_4h_aligned[i-1] if not np.isnan(hma_4h_aligned[i-1]) else False
        
        # === 12h CONFIRMATION ===
        price_above_12h = close[i] > hma_12h_aligned[i]
        price_below_12h = close[i] < hma_12h_aligned[i]
        
        # === 15m MOMENTUM (HMA cross) ===
        hma_bullish = hma_8[i] > hma_21[i]
        hma_bearish = hma_8[i] < hma_21[i]
        
        # === RSI (LOOSE thresholds for trade generation) ===
        rsi = rsi_7[i]
        rsi_oversold = rsi < 50  # LOOSE: was <30
        rsi_overbought = rsi > 50  # LOOSE: was >70
        
        # === SMA FILTER (major trend) ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === SESSION FILTER (00-12 UTC preferred but not required) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        is_liquid_session = 0 <= hour_utc <= 12
        
        # === ENTRY LOGIC (LOOSE - MUST generate trades) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + (RSI oversold OR HMA bullish) + price above SMA200
        # Only 2 of 3 conditions needed (very loose)
        long_conditions = 0
        if price_above_4h and hma_4h_slope_bullish:
            long_conditions += 1
        if rsi_oversold:
            long_conditions += 1
        if hma_bullish:
            long_conditions += 1
        if price_above_sma200:
            long_conditions += 1
        
        if long_conditions >= 2 and price_above_4h:
            if rsi_oversold or hma_bullish:
                desired_signal = SIZE_BASE
                if is_liquid_session and price_above_12h:
                    desired_signal = SIZE_STRONG
        
        # SHORT: 4h bearish + (RSI overbought OR HMA bearish) + price below SMA200
        short_conditions = 0
        if price_below_4h and hma_4h_slope_bearish:
            short_conditions += 1
        if rsi_overbought:
            short_conditions += 1
        if hma_bearish:
            short_conditions += 1
        if price_below_sma200:
            short_conditions += 1
        
        if short_conditions >= 2 and price_below_4h:
            if rsi_overbought or hma_bearish:
                desired_signal = -SIZE_BASE
                if is_liquid_session and price_below_12h:
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