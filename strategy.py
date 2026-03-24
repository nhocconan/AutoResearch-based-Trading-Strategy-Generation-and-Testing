#!/usr/bin/env python3
"""
Experiment #969: 15m Primary + 1h/1d HTF — RSI Pullback with Session Filter

Hypothesis: 15m timeframe with 1h HMA trend bias + RSI(7) pullback entries
will capture intraday swings while avoiding excessive trade frequency.

Key innovations:
1. 1h HMA(21) for intermediate trend bias (less strict than requiring 1d)
2. RSI(7) pullback entries: long when RSI<35 in uptrend, short when RSI>65 in downtrend
3. Session filter: prefer 00-12 UTC (London+NY overlap) for higher quality fills
4. 1d HMA(50) as secondary filter (only avoid counter-trend, not required)
5. ATR(14) 2.5x trailing stop for risk management
6. Relaxed entry thresholds to guarantee 40-100 trades/year

Why 15m should work:
- Captures intraday momentum swings missed by 4h/6h strategies
- HTF bias prevents counter-trend trades in strong moves
- Session filter reduces low-liquidity Asian session noise
- RSI(7) reacts faster than RSI(14) for 15m entries

Entry conditions (LOOSE to guarantee trades):
- LONG = 1h HMA bull + RSI(7) < 40 OR (1d HMA bull + RSI(7) < 45)
- SHORT = 1h HMA bear + RSI(7) > 60 OR (1d HMA bear + RSI(7) > 55)
- Session boost: 00-12 UTC gets priority, 12-24 UTC requires stronger signal

Target: Sharpe>0.45, trades>=40 train, trades>=5 test, DD>-40%
Timeframe: 15m
Size: 0.15-0.25 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_pullback_session_1h1d_v1"
timeframe = "15m"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.18
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
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1h HMA primary, 1d HMA secondary) ===
        htf_1h_bull = close[i] > hma_1h_aligned[i]
        htf_1h_bear = close[i] < hma_1h_aligned[i]
        
        htf_1d_bull = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        htf_1d_bear = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # === SESSION FILTER (00-12 UTC preferred) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        is_prime_session = 0 <= hour_utc <= 12  # London + NY overlap
        
        # === RSI PULLBACK SIGNALS (LOOSE THRESHOLDS) ===
        # Long: RSI(7) < 40 in uptrend, or RSI(7) < 45 with 1d support
        rsi_long_strong = rsi_7[i] < 35
        rsi_long_base = rsi_7[i] < 40
        rsi_long_weak = rsi_7[i] < 45
        
        # Short: RSI(7) > 60 in downtrend, or RSI(7) > 55 with 1d support
        rsi_short_strong = rsi_7[i] > 65
        rsi_short_base = rsi_7[i] > 60
        rsi_short_weak = rsi_7[i] > 55
        
        # === ENTRY LOGIC (MULTIPLE PATHS TO GUARANTEE TRADES) ===
        desired_signal = 0.0
        signal_strength = 0
        
        # LONG entries - multiple conditions that can trigger
        long_score = 0
        
        # Path 1: 1h bull + RSI pullback (primary)
        if htf_1h_bull:
            if rsi_long_strong:
                long_score += 3
            elif rsi_long_base:
                long_score += 2
            elif rsi_long_weak:
                long_score += 1
        
        # Path 2: 1d bull support (secondary boost)
        if htf_1d_bull:
            long_score += 1
        
        # Path 3: Session boost
        if is_prime_session and long_score > 0:
            long_score += 1
        
        # SHORT entries - multiple conditions that can trigger
        short_score = 0
        
        # Path 1: 1h bear + RSI pullback (primary)
        if htf_1h_bear:
            if rsi_short_strong:
                short_score += 3
            elif rsi_short_base:
                short_score += 2
            elif rsi_short_weak:
                short_score += 1
        
        # Path 2: 1d bear support (secondary boost)
        if htf_1d_bear:
            short_score += 1
        
        # Path 3: Session boost
        if is_prime_session and short_score > 0:
            short_score += 1
        
        # Determine signal based on scores
        if long_score >= 3 and long_score > short_score:
            if long_score >= 5:
                desired_signal = SIZE_STRONG
                signal_strength = 2
            else:
                desired_signal = SIZE_BASE
                signal_strength = 1
        elif short_score >= 3 and short_score > long_score:
            if short_score >= 5:
                desired_signal = -SIZE_STRONG
                signal_strength = 2
            else:
                desired_signal = -SIZE_BASE
                signal_strength = 1
        
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