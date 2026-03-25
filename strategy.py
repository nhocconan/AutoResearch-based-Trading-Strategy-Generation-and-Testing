#!/usr/bin/env python3
"""
Experiment #1121: 15m Primary + 1h/4h/1d HTF — Fisher Transform + KAMA Trend + Volume Session

Hypothesis: 15m timeframe with Ehlers Fisher Transform for reversal entries, KAMA for adaptive
trend filtering, volume confirmation, and session timing will capture intraday swings while
avoiding noise. This is the FIRST 15m experiment - targeting 50-100 trades/year.

Key innovations:
1. Ehlers Fisher Transform (period=9): Transforms price to Gaussian distribution, crosses at
   extremes (-1.5/+1.5) signal reversals. Works well in bear/range markets.
2. KAMA (Kaufman Adaptive Moving Average, ER=10): Adapts smoothing based on volatility.
   Faster in trends, slower in chop. Better than EMA for crypto.
3. Volume confirmation: Only enter when volume > 1.3x 20-bar average (avoids low-liquidity traps)
4. Session filter: Prefer 00-14 UTC (London+NY overlap for crypto liquidity)
5. HTF bias: 4h HMA(21) for trend direction, 1d HMA(21) for long-term bias
6. Discrete sizing: 0.0, ±0.15, ±0.20 (smaller size for 15m frequency)
7. ATR(14) 2.5x trailing stop

Why 15m might work:
- Captures intraday swings that 4h/6h miss
- Fisher Transform catches reversals at extremes (proven in bear markets)
- KAMA adapts to crypto's changing volatility regime
- Volume + session filters reduce false signals
- HTF bias prevents counter-trend trades

Entry conditions (LOOSE to guarantee trades):
- LONG: Fisher crosses above -1.2 + price>KAMA + volume>1.3x + 4h_HMA bullish + session OK
- SHORT: Fisher crosses below +1.2 + price<KAMA + volume>1.3x + 4h_HMA bearish + session OK

Target: Sharpe>0.45, trades>=40 train, trades>=5 test, DD>-40%, 50-100 trades/year
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_fisher_kama_volume_session_4h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs noise)
    ER (Efficiency Ratio) = |price change| / sum of |individual changes|
    High ER = trending (use fast SC), Low ER = choppy (use slow SC)
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 1e-10:
            er[i] = price_change / noise
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    sc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = er[i] * (fast_sc - slow_sc) + slow_sc
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] ** 2 * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Transforms price to Gaussian distribution for clearer reversal signals
    Formula: Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = normalized price
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate median price
    median = (high + low + close) / 3.0
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        price_range = hh - ll
        if price_range < 1e-10:
            continue
        
        # Normalize price to -1 to +1 range
        x = (2.0 * median[i] - hh - ll) / price_range
        
        # Clamp to avoid log errors
        x = np.clip(x, -0.999, 0.999)
        
        # Calculate Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        if i > period:
            fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

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

def calculate_volume_avg(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def is_session_active(open_time, start_hour=0, end_hour=14):
    """Check if bar is within preferred trading session (UTC)"""
    # open_time is in milliseconds
    hour = pd.to_datetime(open_time, unit='ms').hour
    return start_hour <= hour < end_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    kama_21 = calculate_kama(close, period=21, fast_period=2, slow_period=30)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    
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
    
    # Track Fisher crosses
    prev_fisher_long_signal = False
    prev_fisher_short_signal = False
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_21[i]) or np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] <= 1e-10:
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
        
        # === HTF BIAS ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong bias when 4h and 1d align
        strong_bull = hma_4h_bull and hma_1d_bull
        strong_bear = hma_4h_bear and hma_1d_bear
        
        # === VOLUME FILTER ===
        volume_spike = volume[i] > 1.3 * vol_avg_20[i]
        
        # === SESSION FILTER ===
        session_ok = is_session_active(open_time[i], start_hour=0, end_hour=14)
        
        # === KAMA TREND FILTER ===
        kama_bull = close[i] > kama_21[i]
        kama_bear = close[i] < kama_21[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.2 from below
        fisher_long_cross = (fisher_prev[i] < -1.2) and (fisher[i] >= -1.2)
        # Short: Fisher crosses below +1.2 from above
        fisher_short_cross = (fisher_prev[i] > 1.2) and (fisher[i] <= 1.2)
        
        # === ENTRY LOGIC (LOOSE CONDITIONS TO GENERATE TRADES) ===
        desired_signal = 0.0
        
        # LONG entry: Fisher cross + KAMA bull + volume + (HTF bias OR session)
        if fisher_long_cross:
            if kama_bull and volume_spike:
                # Strong signal with HTF alignment
                if strong_bull:
                    desired_signal = SIZE_STRONG
                # Base signal with session filter
                elif session_ok:
                    desired_signal = SIZE_BASE
                # Weaker signal without session but with HTF
                elif hma_4h_bull:
                    desired_signal = SIZE_BASE
        
        # SHORT entry: Fisher cross + KAMA bear + volume + (HTF bias OR session)
        elif fisher_short_cross:
            if kama_bear and volume_spike:
                # Strong signal with HTF alignment
                if strong_bear:
                    desired_signal = -SIZE_STRONG
                # Base signal with session filter
                elif session_ok:
                    desired_signal = -SIZE_BASE
                # Weaker signal without session but with HTF
                elif hma_4h_bear:
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