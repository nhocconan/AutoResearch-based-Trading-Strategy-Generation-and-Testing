#!/usr/bin/env python3
"""
Experiment #956: 30m Primary + 4h/1d HTF — ATR Compression + RSI Pullback

Hypothesis: 30m timeframe with volatility compression detection + HTF trend bias
will generate fewer but higher-quality trades than pure trend strategies.

Key innovations:
1. ATR Compression Ratio: ATR(7)/ATR(30) < 0.6 detects volatility squeeze before breakout
2. 4h HMA(21) for intermediate trend direction (proven in best strategy)
3. 1d SMA(200) for long-term regime filter
4. RSI(14) pullback entries in direction of HTF trend
5. Session filter: 08-20 UTC only (reduces overnight noise trades)
6. ATR(14) 2.5x trailing stop for risk management

Why this should work:
- ATR compression catches breakouts after consolidation (low fee churn)
- 4h HMA provides proven trend bias from best strategy
- Session filter reduces trades to target 40-80/year
- RSI pullback entries have better risk/reward than breakouts
- 30m captures intraday swings without 15m noise

Entry conditions (balanced for trades + quality):
- LONG = 4h bull + 1d bull + ATR_compressed + RSI(14)<45 + session
- SHORT = 4h bear + 1d bear + ATR_compressed + RSI(14)>55 + session

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 30m
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_atr_compress_rsi_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    
    return sma

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = ((open_time_array // 1000) // 3600) % 24
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
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    sma_1d_raw = calculate_sma(df_1d['close'].values, period=200)
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # Calculate 30m indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # ATR Compression Ratio
    atr_ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(atr_7[i]) and not np.isnan(atr_30[i]) and atr_30[i] > 1e-10:
            atr_ratio[i] = atr_7[i] / atr_30[i]
    
    # Session filter: 08-20 UTC only
    utc_hours = get_hour_from_open_time(open_time)
    in_session = (utc_hours >= 8) & (utc_hours <= 20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h HMA + 1d SMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_1d_bull = close[i] > sma_1d_aligned[i]
        htf_1d_bear = close[i] < sma_1d_aligned[i]
        
        # === ATR COMPRESSION (volatility squeeze) ===
        is_compressed = atr_ratio[i] < 0.65  # Volatility compression
        
        # === RSI PULLBACK (not extreme, just pullback) ===
        rsi_pullback_long = rsi_14[i] < 45  # Pullback in uptrend
        rsi_pullback_short = rsi_14[i] > 55  # Pullback in downtrend
        
        # === SESSION FILTER ===
        is_session = in_session[i]
        
        # === ENTRY LOGIC (3+ CONFLUENCE) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 1d bull + ATR compressed + RSI pullback + session
        if htf_4h_bull and htf_1d_bull:
            if is_compressed and rsi_pullback_long and is_session:
                desired_signal = SIZE_BASE
            # Stronger signal if RSI very oversold
            elif is_compressed and rsi_14[i] < 35 and is_session:
                desired_signal = SIZE_STRONG
        
        # SHORT: 4h bear + 1d bear + ATR compressed + RSI pullback + session
        elif htf_4h_bear and htf_1d_bear:
            if is_compressed and rsi_pullback_short and is_session:
                desired_signal = -SIZE_BASE
            # Stronger signal if RSI very overbought
            elif is_compressed and rsi_14[i] > 65 and is_session:
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