#!/usr/bin/env python3
"""
Experiment #859: 1h Primary + 4h/12h HTF — HMA Trend + RSI + Choppiness + Session

Hypothesis: 1h timeframe with 4h/12h HTF bias can achieve optimal trade frequency
(40-80 trades/year) with regime-adaptive logic. Key is using HTF for DIRECTION,
1h only for ENTRY TIMING. Session filter (08-20 UTC) reduces noise during low-volume
Asian hours. Choppiness Index switches between trend-follow and mean-revert modes.

Key innovations:
1. 4h HMA(21) for primary HTF trend bias (proven in best strategies)
2. 12h HMA(21) for secondary confirmation (smoother than 4h)
3. 1h HMA(16/48) dual crossover for entry timing
4. RSI(14) extremes for mean-reversion entries in range regime
5. Choppiness(14) regime switch: >50 = range, <50 = trend
6. Session filter: only trade 08-20 UTC (high volume hours)
7. ATR(14) 2.5x trailing stop for risk management
8. Discrete sizing: 0.0, ±0.20, ±0.25 to minimize fee churn

Entry conditions (LOOSE to ensure ≥30 trades/train, ≥3/test):
- TREND REGIME (CHOP<50, HTF bull): LONG = 1h HMA16>48 OR RSI<45
- TREND REGIME (CHOP<50, HTF bear): SHORT = 1h HMA16<48 OR RSI>55
- RANGE REGIME (CHOP>50, HTF bull): LONG = RSI<35 (oversold)
- RANGE REGIME (CHOP>50, HTF bear): SHORT = RSI>65 (overbought)
- SESSION: Only enter 08-20 UTC (avoid Asian session noise)

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 1h
Size: 0.20-0.25 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_chop_session_4h12h_v1"
timeframe = "1h"
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
    
    # WMA helper
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
    # WMA of diff with sqrt(n)
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Using 50 as threshold for regime switch
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def get_hour_from_open_time(open_time_col):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = (open_time_col // (1000 * 3600)) % 24
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
    
    # Calculate and align HTF HMA
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    hma_1h_16 = calculate_hma(close, period=16)
    hma_1h_48 = calculate_hma(close, period=48)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Extract UTC hour for session filter
    utc_hours = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
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
        
        if np.isnan(hma_1h_16[i]) or np.isnan(hma_1h_48[i]) or np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
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
        
        # === SESSION FILTER (08-20 UTC only) ===
        in_session = (utc_hours[i] >= 8) and (utc_hours[i] <= 20)
        
        # === HTF BIAS (4h + 12h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        # Combined HTF bias (both must agree for strong signal)
        htf_bull = htf_4h_bull and htf_12h_bull
        htf_bear = htf_4h_bear and htf_12h_bear
        htf_neutral = not htf_bull and not htf_bear
        
        # === 1h HMA CROSSOVER ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_1h_16[i-1]) and not np.isnan(hma_1h_48[i-1]):
            hma_crossover_long = (hma_1h_16[i-1] <= hma_1h_48[i-1]) and (hma_1h_16[i] > hma_1h_48[i])
            hma_crossover_short = (hma_1h_16[i-1] >= hma_1h_48[i-1]) and (hma_1h_16[i] < hma_1h_48[i])
        
        # === HMA TREND ===
        hma_1h_bull = hma_1h_16[i] > hma_1h_48[i]
        hma_1h_bear = hma_1h_16[i] < hma_1h_48[i]
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_weak_long = rsi_14[i] < 45.0
        rsi_weak_short = rsi_14[i] > 55.0
        
        # === CHOPPINESS REGIME ===
        chop_trending = chop_14[i] < 50.0
        chop_ranging = chop_14[i] >= 50.0
        
        # === ENTRY LOGIC (REGIME ADAPTIVE + LOOSE FOR TRADES) ===
        desired_signal = 0.0
        
        if not in_session:
            # Outside session hours - flat or maintain existing position
            if in_position:
                desired_signal = position_side * SIZE_BASE
            else:
                desired_signal = 0.0
        else:
            # In session - can enter new positions
            if htf_bull:
                # Bullish HTF bias
                if chop_trending:
                    # Trend regime: use HMA crossover or weak RSI
                    if hma_crossover_long:
                        desired_signal = SIZE_STRONG
                    elif hma_1h_bull or rsi_weak_long:
                        desired_signal = SIZE_BASE
                else:
                    # Range regime: use RSI oversold for mean reversion
                    if rsi_oversold:
                        desired_signal = SIZE_STRONG
                    elif rsi_weak_long:
                        desired_signal = SIZE_BASE
            
            elif htf_bear:
                # Bearish HTF bias
                if chop_trending:
                    # Trend regime: use HMA crossover or weak RSI
                    if hma_crossover_short:
                        desired_signal = -SIZE_STRONG
                    elif hma_1h_bear or rsi_weak_short:
                        desired_signal = -SIZE_BASE
                else:
                    # Range regime: use RSI overbought for mean reversion
                    if rsi_overbought:
                        desired_signal = -SIZE_STRONG
                    elif rsi_weak_short:
                        desired_signal = -SIZE_BASE
            
            else:
                # HTF neutral - use 1h signals only (loose conditions)
                if chop_trending:
                    if hma_crossover_long:
                        desired_signal = SIZE_BASE
                    elif hma_crossover_short:
                        desired_signal = -SIZE_BASE
                else:
                    if rsi_oversold:
                        desired_signal = SIZE_BASE
                    elif rsi_overbought:
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