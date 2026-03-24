#!/usr/bin/env python3
"""
Experiment #976: 30m Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: 30m timeframe with 4h HMA trend bias + 1d momentum filter + RSI pullback entries
will generate consistent trades (40-80/year) with positive Sharpe on ALL symbols.

Key innovations:
1. 4h HMA(21) for intermediate trend direction (HTF bias)
2. 1d momentum (close > SMA50) for daily bias confirmation
3. 30m RSI(14) pullback entries: long when RSI<45 in uptrend, short when RSI>55 in downtrend
4. Session filter: 08-20 UTC (London/NY overlap = cleaner moves, less noise)
5. Choppiness Index(14) regime filter: avoid entries when CHOP>61.8 (too choppy)
6. ATR(14) 2.5x trailing stop for risk management

Why this should work:
- 4h HMA provides stable trend bias (less noise than 30m)
- RSI pullback (45/55) is LOOSE enough to generate trades (unlike 30/70)
- Session filter reduces false signals during low-volume Asian hours
- CHOP filter avoids whipsaw in ranging markets
- 30m captures intraday swings without 15m noise or 1h lag

Entry conditions (LOOSE to guarantee trades):
- LONG = 4h HMA bull + 1d bull + RSI(14)<50 + CHOP<61.8 + session active
- SHORT = 4h HMA bear + 1d bear + RSI(14)>50 + CHOP<61.8 + session active
- Relaxed RSI thresholds (50 instead of 40/60) for more trades

Target: Sharpe>0.45, trades>=40 train, trades>=5 test, DD>-40%
Timeframe: 30m
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_session_chop_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = range/choppy (avoid trend entries)
    CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
        
        if highest_high > lowest_low and tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10((highest_high - lowest_low) / tr_sum) / np.log10(period)
    
    return chop

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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    sma_1d_raw = calculate_sma(df_1d['close'].values, period=50)
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # Calculate 30m indicators
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
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
    
    for i in range(150, n):
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
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        session_active = (hour_utc >= 8) and (hour_utc <= 20)
        
        # === HTF BIAS (4h HMA + 1d SMA50) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_1d_bull = close[i] > sma_1d_aligned[i]
        htf_1d_bear = close[i] < sma_1d_aligned[i]
        
        # === REGIME FILTER (CHOP) ===
        is_choppy = chop_14[i] > 61.8  # Avoid entries in choppy markets
        
        # === RSI PULLBACK (LOOSE THRESHOLDS FOR TRADES) ===
        rsi_oversold = rsi_14[i] < 50  # Relaxed from 40
        rsi_overbought = rsi_14[i] > 50  # Relaxed from 60
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG entries: HTF bull + RSI pullback + session + not choppy
        if htf_4h_bull and htf_1d_bull and not is_choppy and session_active and rsi_oversold:
            desired_signal = SIZE_BASE
            # Strong signal if RSI very oversold
            if rsi_14[i] < 35:
                desired_signal = SIZE_STRONG
        
        # SHORT entries: HTF bear + RSI pullback + session + not choppy
        elif htf_4h_bear and htf_1d_bear and not is_choppy and session_active and rsi_overbought:
            desired_signal = -SIZE_BASE
            # Strong signal if RSI very overbought
            if rsi_14[i] > 65:
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