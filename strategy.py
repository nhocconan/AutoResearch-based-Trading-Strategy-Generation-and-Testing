#!/usr/bin/env python3
"""
Experiment #1616: 30m Primary + 4h/1d HTF — Fisher Transform + HMA Trend Bias

Hypothesis: After 13+ failed experiments with overly strict entry conditions (0 trades),
this strategy uses LOOSE Fisher Transform thresholds to GUARANTEE trade generation.
Key insight from failures: adding too many confluence filters = 0 trades.

Innovations vs failed 30m/15m/5m attempts (#1605, #1610, #1613):
1. FISHER TRANSFORM (period=9): Better reversal capture than RSI in bear/range markets
2. LOOSE THRESHOLDS: Fisher -1.0/+1.0 (not -1.5/-2.0) to ensure ≥40 trades/year
3. HTF AS BIAS NOT FILTER: 4h HMA increases size but doesn't block entries
4. SESSION AS ENHANCER: 08-20 UTC increases size, not entry requirement
5. SINGLE PRIMARY SIGNAL: Fisher cross is the trigger, everything else modifies size

Why this should work where others failed:
- Fisher proven superior to RSI for reversal timing (Ehlers research)
- Loose thresholds guarantee trades (learning from 0-trade failures)
- 30m TF with 4h bias = fewer trades than pure 30m, more than 1h/4h
- Session filter captures liquid hours without blocking entries

Entry logic (LOOSE to guarantee ≥40 trades/year):
- LONG: Fisher crosses above -1.0 (always valid), 4h HMA bullish increases size
- SHORT: Fisher crosses below +1.0 (always valid), 4h HMA bearish increases size
- Session 08-20 UTC: +50% size boost (liquidity confirmation)

Target: Sharpe>0.6, trades>=40/year train, trades>=5 test, DD>-35%
Timeframe: 30m
Size: 0.20 base, 0.30 with HTF confirm, 0.35 with session
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_hma_bias_session_loose_v1"
timeframe = "30m"
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

def calculate_fisher(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Better at catching extremes than RSI, especially in bear markets
    Returns fisher value and trigger (previous value for crossover detection)
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    trigger = np.full(n, np.nan, dtype=np.float64)
    
    median = (high + low) / 2.0
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        price_range = highest - lowest
        
        if price_range < 1e-10:
            fisher[i] = 0.0
            if i > 0:
                trigger[i] = fisher[i-1]
            continue
        
        normalized = 2.0 * (median[i] - lowest) / price_range - 1.0
        normalized = max(-0.999, min(0.999, normalized))
        
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        if i > 0:
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

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

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    return (open_time // 3600000) % 24

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
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    fisher, fisher_trigger = calculate_fisher(high, low, period=9)
    
    signals = np.zeros(n)
    
    # Position sizing levels (discrete to minimize fee churn)
    SIZE_BASE = 0.20      # Base position
    SIZE_HTF = 0.28       # With HTF confirmation
    SIZE_SESSION = 0.35   # With session + HTF
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === FISHER TRANSFORM SIGNALS (LOOSE THRESHOLDS) ===
        fisher_val = fisher[i]
        fisher_prev = fisher_trigger[i]
        
        # Fisher crossover signals - LOOSE for trade generation
        fisher_bull_cross = fisher_val > -1.0 and fisher_prev <= -1.0
        fisher_bear_cross = fisher_val < 1.0 and fisher_prev >= 1.0
        
        # Fisher extreme levels for additional confirmation
        fisher_oversold = fisher_val < -0.5
        fisher_overbought = fisher_val > 0.5
        
        # === HTF TREND BIAS (4h HMA) ===
        hma_4h_valid = not np.isnan(hma_4h_aligned[i])
        hma_1d_valid = not np.isnan(hma_1d_aligned[i])
        
        bullish_4h = hma_4h_valid and close[i] > hma_4h_aligned[i]
        bearish_4h = hma_4h_valid and close[i] < hma_4h_aligned[i]
        bullish_1d = hma_1d_valid and close[i] > hma_1d_aligned[i]
        bearish_1d = hma_1d_valid and close[i] < hma_1d_aligned[i]
        
        # === SESSION FILTER (08-20 UTC) ===
        hour = get_session_hour(open_time[i])
        is_session = 8 <= hour <= 20
        
        # === RSI FILTER (optional confirmation) ===
        rsi_valid = not np.isnan(rsi_14[i])
        rsi_neutral = rsi_valid and 35 < rsi_14[i] < 65
        
        # === ENTRY LOGIC (LOOSE - MUST GENERATE TRADES) ===
        desired_signal = 0.0
        signal_strength = 0  # 0=base, 1=htf, 2=session
        
        # LONG entries - Fisher bull cross is primary trigger
        if fisher_bull_cross:
            signal_strength = 0  # Base size always valid on Fisher cross
            
            # Upgrade size with HTF confirmation
            if bullish_4h or bullish_1d:
                signal_strength = 1
            
            # Upgrade further with session
            if is_session and (bullish_4h or bullish_1d):
                signal_strength = 2
            
            # Also allow long if deeply oversold even without cross
            if fisher_oversold and fisher_val < fisher_prev:
                if signal_strength < 1 and (bullish_4h or bullish_1d):
                    signal_strength = 1
            
            if signal_strength == 2:
                desired_signal = SIZE_SESSION
            elif signal_strength == 1:
                desired_signal = SIZE_HTF
            else:
                desired_signal = SIZE_BASE
        
        # SHORT entries - Fisher bear cross is primary trigger
        elif fisher_bear_cross:
            signal_strength = 0  # Base size always valid on Fisher cross
            
            # Upgrade size with HTF confirmation
            if bearish_4h or bearish_1d:
                signal_strength = 1
            
            # Upgrade further with session
            if is_session and (bearish_4h or bearish_1d):
                signal_strength = 2
            
            # Also allow short if deeply overbought even without cross
            if fisher_overbought and fisher_val > fisher_prev:
                if signal_strength < 1 and (bearish_4h or bearish_1d):
                    signal_strength = 1
            
            if signal_strength == 2:
                desired_signal = -SIZE_SESSION
            elif signal_strength == 1:
                desired_signal = -SIZE_HTF
            else:
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
        if desired_signal >= SIZE_SESSION * 0.9:
            final_signal = SIZE_SESSION
        elif desired_signal <= -SIZE_SESSION * 0.9:
            final_signal = -SIZE_SESSION
        elif desired_signal >= SIZE_HTF * 0.9:
            final_signal = SIZE_HTF
        elif desired_signal <= -SIZE_HTF * 0.9:
            final_signal = -SIZE_HTF
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