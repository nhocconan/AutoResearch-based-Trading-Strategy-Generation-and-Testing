#!/usr/bin/env python3
"""
Experiment #1057: 15m Primary + 4h/12h HTF — Session RSI Pullback with Choppiness Filter

Hypothesis: 15m strategies fail due to either (a) too many trades → fee drag, or (b) too strict entries → 0 trades.
This strategy uses 4h HMA for trend direction, 15m RSI(7) for fast pullback entries, UTC session filter (00-12),
and 4h Choppiness to avoid range markets. LOOSE entry conditions guarantee trades while session filter controls frequency.

Key innovations:
1. 4h HMA(21) for trend bias — simpler than triple HMA, faster computation
2. 15m RSI(7) for entries — faster than RSI(14), catches intraday pullbacks
3. UTC Session filter (00-12) — London+NY overlap, reduces trades by ~50%
4. 4h Choppiness > 50 filter — avoid choppy 4h markets (whipsaw protection)
5. LOOSE entries: RSI(7)<40 for long in uptrend, RSI(7)>60 for short in downtrend
6. Smaller sizing (0.15-0.25) for 15m frequency — target 50-80 trades/year
7. ATR(14) 2.0x stoploss — tighter for lower timeframe

Why this should work on 15m:
- 4h trend filter prevents counter-trend trades (major killer on 15m)
- RSI(7) is responsive enough for 15m pullbacks without being noisy
- Session filter cuts trade count in half (critical for fee drag)
- Choppiness filter avoids 4h range markets where 15m gets chopped
- LOOSE entries ensure we actually generate trades (learning from 0-trade failures)

Entry conditions (LOOSE to guarantee ≥10 trades/symbol):
- LONG: 4h_CHOP<50 + 4h_HMA_bull + 15m_RSI(7)<40 + session 00-12 UTC
- SHORT: 4h_CHOP<50 + 4h_HMA_bear + 15m_RSI(7)>60 + session 00-12 UTC

Target: Sharpe>0.45, trades>=40/symbol train, trades>=5/symbol test, DD>-35%
Timeframe: 15m
Size: 0.15-0.25 discrete (smaller for 15m frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_session_rsi_pullback_4h_chop_v1"
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = ranging market, CHOP < 38.2 = trending market
    We use threshold of 50 for filter
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    chop_4h_raw = calculate_choppiness(
        df_4h['high'].values,
        df_4h['low'].values,
        df_4h['close'].values,
        period=14
    )
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(chop_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (UTC 00-12 only) ===
        # open_time is in milliseconds since epoch
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = (hour_utc >= 0) and (hour_utc < 12)
        
        # === 4h REGIME FILTER (Choppiness) ===
        is_trending_4h = chop_4h_aligned[i] < 50.0  # Not choppy
        
        # === 4h TREND BIAS (HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === ENTRY LOGIC (LOOSE to guarantee trades) ===
        desired_signal = 0.0
        
        # Only trade in session and when 4h is trending (not choppy)
        if in_session and is_trending_4h:
            # LONG: 4h uptrend + 15m RSI(7) pullback
            if hma_4h_bull and rsi_7[i] < 40.0:
                desired_signal = SIZE_BASE
            # Stronger long signal at more extreme RSI
            elif hma_4h_bull and rsi_7[i] < 30.0:
                desired_signal = SIZE_STRONG
            
            # SHORT: 4h downtrend + 15m RSI(7) bounce
            elif hma_4h_bear and rsi_7[i] > 60.0:
                desired_signal = -SIZE_BASE
            # Stronger short signal at more extreme RSI
            elif hma_4h_bear and rsi_7[i] > 70.0:
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