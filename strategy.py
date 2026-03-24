#!/usr/bin/env python3
"""
Experiment #909: 15m Primary + 1h/1d HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: 15m timeframe with 1h trend confirmation and 1d regime bias can capture
intraday moves while avoiding whipsaw. Key insight from 15m failures: entry conditions
must be LOOSE enough to generate trades (≥10/train, ≥3/test) while HTF filters maintain
signal quality. Previous 15m strategies failed with Sharpe=0.000 (zero trades).

Innovations:
1. 1d HMA(21) for long-term regime bias (bull/bear market)
2. 1h HMA(16/48) dual crossover for intermediate trend confirmation
3. 15m RSI(7) for entry timing — LOOSE thresholds (40/60 not 30/70)
4. Session filter: 00-12 UTC gets full size, 12-24 UTC gets half size
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.20, ±0.25 (smaller for 15m frequency)

Entry logic (LOOSE to ensure trades):
- LONG: 1d HMA bull + 1h HMA(16)>HMA(48) + RSI(7)<45
- SHORT: 1d HMA bear + 1h HMA(16)<HMA(48) + RSI(7)>55
- Session boost: 00-12 UTC = full size, 12-24 UTC = half size

Target: Sharpe>0.45, trades>=10 train, trades>=3 test, DD>-40%, 40-100 trades/year
Timeframe: 15m
Size: 0.20-0.25 discrete (smaller than higher TF due to frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_session_1h1d_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    return (open_time // (1000 * 3600)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1h_16_raw = calculate_hma(df_1h['close'].values, period=16)
    hma_1h_16_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_16_raw)
    
    hma_1h_48_raw = calculate_hma(df_1h['close'].values, period=48)
    hma_1h_48_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_48_raw)
    
    # Calculate 15m indicators
    hma_15m_16 = calculate_hma(close, period=16)
    hma_15m_48 = calculate_hma(close, period=48)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    SIZE_SESSION_HALF = 0.10
    
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
        
        if np.isnan(hma_15m_16[i]) or np.isnan(hma_15m_48[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1h_16_aligned[i]) or np.isnan(hma_1h_48_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 1h HMA TREND ===
        htf_1h_bull = hma_1h_16_aligned[i] > hma_1h_48_aligned[i]
        htf_1h_bear = hma_1h_16_aligned[i] < hma_1h_48_aligned[i]
        
        # === 15m HMA TREND ===
        hma_15m_bull = hma_15m_16[i] > hma_15m_48[i]
        hma_15m_bear = hma_15m_16[i] < hma_15m_48[i]
        
        # === RSI CONDITIONS (LOOSE for trade generation) ===
        rsi_oversold = rsi_7[i] < 45.0  # Long entry
        rsi_overbought = rsi_7[i] > 55.0  # Short entry
        rsi_neutral_long = rsi_7[i] < 50.0
        rsi_neutral_short = rsi_7[i] > 50.0
        
        # === SESSION FILTER ===
        utc_hour = get_utc_hour(open_time[i])
        is_peak_session = 0 <= utc_hour < 12  # London/NY overlap
        session_multiplier = 1.0 if is_peak_session else 0.5
        
        # === ENTRY LOGIC (LOOSE to ensure trades) ===
        desired_signal = 0.0
        
        # LONG conditions: 1d bull + 1h bull + RSI pullback
        if htf_1d_bull and htf_1h_bull:
            if rsi_oversold:
                desired_signal = SIZE_STRONG * session_multiplier
            elif rsi_neutral_long and hma_15m_bull:
                desired_signal = SIZE_BASE * session_multiplier
        
        # SHORT conditions: 1d bear + 1h bear + RSI pushback
        elif htf_1d_bear and htf_1h_bear:
            if rsi_overbought:
                desired_signal = -SIZE_STRONG * session_multiplier
            elif rsi_neutral_short and hma_15m_bear:
                desired_signal = -SIZE_BASE * session_multiplier
        
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
        if desired_signal >= SIZE_STRONG * 0.8:
            final_signal = SIZE_STRONG if is_peak_session else SIZE_BASE
        elif desired_signal <= -SIZE_STRONG * 0.8:
            final_signal = -SIZE_STRONG if is_peak_session else -SIZE_BASE
        elif desired_signal >= SIZE_BASE * 0.8:
            final_signal = SIZE_BASE if is_peak_session else SIZE_SESSION_HALF
        elif desired_signal <= -SIZE_BASE * 0.8:
            final_signal = -SIZE_BASE if is_peak_session else -SIZE_SESSION_HALF
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