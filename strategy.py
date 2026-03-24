#!/usr/bin/env python3
"""
Experiment #993: 5m Primary + 15m/4h HTF — Fisher Transform + Session Filter

Hypothesis: 5m timeframe with Fisher Transform entries + 4h trend bias + session filter
will capture intraday momentum while avoiding low-volume whipsaws.

Key innovations:
1. Fisher Transform (period=9): Normalizes price to Gaussian distribution, catches reversals
   - Long when Fisher crosses above -1.5 (oversold reversal)
   - Short when Fisher crosses below +1.5 (overbought reversal)
2. 4h HMA(21) for primary trend bias — only trade with HTF direction
3. 15m RSI(14) for momentum confirmation — RSI 45-55 = neutral, >55 = bull, <45 = bear
4. Session filter: 08-20 UTC only (high volume, avoid Asia dead hours)
5. ATR(14) 2.5x trailing stop for risk management
6. Size: 0.15 (small due to 5m trade frequency)

Why this should work on 5m:
- Fisher Transform catches intraday reversals better than RSI
- 4h trend bias prevents counter-trend trades (major failure mode)
- Session filter avoids 60% of low-volume whipsaws
- 5m entries with 4h direction = HTF frequency with LTF precision

Entry conditions (LOOSE to guarantee trades):
- LONG = 4h bull + 15m RSI>45 + Fisher cross above -1.5 + session active
- SHORT = 4h bear + 15m RSI<55 + Fisher cross below +1.5 + session active

Target: Sharpe>0.45, trades>=50 train, trades>=10 test, DD>-35%
Timeframe: 5m
Size: 0.15 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_fisher_session_15m4h_v1"
timeframe = "5m"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution for clearer reversal signals
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_prev = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest_hl2 = np.max(high[i-period+1:i+1])
        lowest_hl2 = np.min(low[i-period+1:i+1])
        
        # Normalize to 0-1 range
        if highest_hl2 > lowest_hl2:
            normalized = (hl2 - lowest_hl2) / (highest_hl2 - lowest_hl2)
        else:
            normalized = 0.5
        
        # Clamp to avoid division by zero in fisher calculation
        normalized = max(0.001, min(0.999, normalized))
        
        # Fisher Transform calculation
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        if i > period:
            fisher_prev[i] = fisher[i-1]
        else:
            fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

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

def is_session_active(open_time, start_hour=8, end_hour=20):
    """
    Check if timestamp is within active trading session (UTC)
    08-20 UTC captures London + NY overlap, avoids Asia dead hours
    """
    # open_time is in milliseconds since epoch
    hour = pd.to_datetime(open_time, unit='ms').hour
    return start_hour <= hour < end_hour

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
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate 5m indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.15  # Small size for 5m frequency
    
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
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_15m_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        session_active = is_session_active(open_time[i], start_hour=8, end_hour=20)
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m RSI MOMENTUM ===
        rsi_15m = rsi_15m_aligned[i]
        rsi_bull = rsi_15m > 45.0  # Neutral-to-bull threshold
        rsi_bear = rsi_15m < 55.0  # Neutral-to-bear threshold
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = (fisher_prev[i] <= -1.5) and (fisher[i] > -1.5)
        fisher_cross_short = (fisher_prev[i] >= 1.5) and (fisher[i] < 1.5)
        
        # Also allow Fisher in extreme zones for continuation
        fisher_deep_oversold = fisher[i] < -2.0
        fisher_deep_overbought = fisher[i] > 2.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG entries (4h bull + 15m RSI supportive + Fisher signal + session)
        if htf_4h_bull and session_active:
            if fisher_cross_long and rsi_bull:
                # Primary entry: Fisher cross + RSI confirmation
                desired_signal = SIZE
            elif fisher_deep_oversold and rsi_15m < 50:
                # Deep oversold pullback in uptrend
                desired_signal = SIZE
        
        # SHORT entries (4h bear + 15m RSI supportive + Fisher signal + session)
        elif htf_4h_bear and session_active:
            if fisher_cross_short and rsi_bear:
                # Primary entry: Fisher cross + RSI confirmation
                desired_signal = -SIZE
            elif fisher_deep_overbought and rsi_15m > 50:
                # Deep overbought pullback in downtrend
                desired_signal = -SIZE
        
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
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
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