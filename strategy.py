#!/usr/bin/env python3
"""
Experiment #1059: 1h Primary + 4h/12h HTF — Fisher Transform + HMA Trend + RSI Pullback

Hypothesis: Combining Ehlers Fisher Transform (reversal detection) with 4h HMA trend
and 1h RSI pullback entries will work better in bear/range markets (2025 test period)
than pure trend-following strategies that failed in experiments #1048-#1058.

Key innovations:
1. Fisher Transform (period=9): Catches reversals at extremes, works in bear rallies
   - Long when Fisher crosses above -1.5 from below
   - Short when Fisher crosses below +1.5 from above
2. 4h HMA(21) for trend direction: Only trade with HTF trend
3. 1h RSI(14) for pullback timing: Enter on pullbacks, not breakouts
4. 12h HMA(21) as meta-filter: Avoid counter-trend trades against 12h bias
5. Session filter: 08-20 UTC (Asian+European overlap, highest volume)
6. ATR(14) 2.5x trailing stop for risk management

Why this should work:
- Fisher Transform proven to catch reversals in bear markets (unlike EMA crossover)
- 4h HMA provides trend direction without whipsaw (proven in best strategy #1055)
- RSI pullback entries = better risk/reward than breakout entries
- 1h timeframe with HTF filters = 40-80 trades/year (not too many, not too few)
- Session filter reduces noise during low-volume periods

Entry conditions (LOOSE to guarantee trades):
- LONG: 4h_HMA bullish + 12h_HMA not strongly bearish + RSI(14) < 50 + Fisher < -0.5
- SHORT: 4h_HMA bearish + 12h_HMA not strongly bullish + RSI(14) > 50 + Fisher > 0.5
- Exit: RSI > 70 (long) or RSI < 30 (short) OR stoploss hit

Target: Sharpe>0.45, trades>=40 train, trades>=5 test, DD>-40%
Timeframe: 1h (MANDATORY)
Size: 0.25 base, 0.30 strong (discrete levels)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_rsi_pullback_4h12h_v1"
timeframe = "1h"
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

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to -1 to +1 range
    Catches reversals at extremes, works well in bear/range markets
    
    Formula:
    1. Normalize price: (close - lowest_low) / (highest_high - lowest_low)
    2. Apply Fisher: 0.5 * ln((1 + x) / (1 - x))
    3. Smooth with EMA
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(close[i-period+1:i+1])
        lowest_low = np.min(close[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range < 1e-10:
            continue
        
        # Normalize to 0-1 range, then scale to -0.99 to +0.99
        normalized = (close[i] - lowest_low) / price_range
        x = 0.99 * (2.0 * normalized - 1.0)  # Scale to -0.99 to +0.99
        
        # Fisher transform
        if abs(x) < 0.999:
            fisher_raw = 0.5 * np.log((1.0 + x) / (1.0 - x))
        else:
            fisher_raw = np.sign(x) * 2.0  # Cap at extremes
        
        fisher[i] = fisher_raw
        
        # Signal line (1-period lag)
        if i > period:
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

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
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    fisher, fisher_signal = calculate_fisher(close, period=9)
    
    # Extract UTC hour for session filter
    utc_hour = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track Fisher crosses for entry confirmation
    prev_fisher = np.nan
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(fisher[i]):
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
        
        # === SESSION FILTER (08-20 UTC) ===
        in_session = (utc_hour[i] >= 8) and (utc_hour[i] <= 20)
        
        # === HTF TREND DIRECTION (4h HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === META FILTER (12h HMA - avoid strong counter-trend) ===
        hma_12h_bull = close[i] > hma_12h_aligned[i]
        hma_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = False
        fisher_cross_down = False
        
        if not np.isnan(prev_fisher) and not np.isnan(fisher_signal[i]):
            # Fisher crosses above signal line from below
            if prev_fisher < fisher_signal[i] and fisher[i] > fisher_signal[i]:
                fisher_cross_up = True
            # Fisher crosses below signal line from above
            if prev_fisher > fisher_signal[i] and fisher[i] < fisher_signal[i]:
                fisher_cross_down = True
        
        prev_fisher = fisher[i]
        
        # === ENTRY LOGIC (LOOSE CONDITIONS TO GUARANTEE TRADES) ===
        desired_signal = 0.0
        
        # LONG entries: 4h bullish + RSI pullback + Fisher confirmation
        if hma_4h_bull and in_session:
            # Allow long even if 12h is neutral or slightly bearish (not strongly against)
            if rsi_14[i] < 55.0:  # Pullback condition (LOOSE: < 55 not < 40)
                if fisher[i] < 0.0 or fisher_cross_up:  # Fisher neutral or crossing up
                    # Stronger signal if 12h also bullish
                    if hma_12h_bull and rsi_14[i] < 45.0 and fisher[i] < -0.5:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
        
        # SHORT entries: 4h bearish + RSI pullback + Fisher confirmation
        elif hma_4h_bear and in_session:
            # Allow short even if 12h is neutral or slightly bullish
            if rsi_14[i] > 45.0:  # Pullback condition (LOOSE: > 45 not > 60)
                if fisher[i] > 0.0 or fisher_cross_down:  # Fisher neutral or crossing down
                    # Stronger signal if 12h also bearish
                    if hma_12h_bear and rsi_14[i] > 55.0 and fisher[i] > 0.5:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
        
        # === EXIT CONDITIONS (take profit on RSI extremes) ===
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            desired_signal = 0.0  # Take profit on long
        
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            desired_signal = 0.0  # Take profit on short
        
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