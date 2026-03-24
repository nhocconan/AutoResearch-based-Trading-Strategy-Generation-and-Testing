#!/usr/bin/env python3
"""
Experiment #1085: 15m Primary + 4h/1d HTF — RSI Mean Reversion with Trend Filter

Hypothesis: 15m timeframe has failed 8+ times due to overly strict entry conditions.
This strategy uses LOOSE entry conditions to GUARANTEE trades while maintaining
quality through HTF trend filter. Key insight: 15m needs fewer confluence factors
than 4h/6h strategies to actually generate signals.

Key innovations:
1. 4h HMA(21) for trend direction — call ONCE before loop (Rule 1)
2. 15m RSI(7) — faster than RSI(14), catches more intraday reversals
3. Bollinger Band position — enter near bands for mean reversion edge
4. Session filter: 00-12 UTC only (London/NY overlap, higher volume)
5. LOOSE entries: RSI(7)<35 or >65 (not extreme 20/80) to guarantee 50+ trades/year
6. ATR(14) 2.5x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.15, ±0.25 to minimize fee churn

Why this should work on 15m:
- RSI(7) is more sensitive than RSI(14) — catches intraday swings
- 4h trend filter prevents counter-trend trades (major edge)
- Session filter reduces trades during low-volume Asia session
- LOOSE conditions guarantee trades (fixes #1 failure mode of 15m strategies)
- Small position size (0.15-0.25) appropriate for higher frequency

Entry conditions (LOOSE to guarantee trades):
- LONG: 4h_HMA bullish + RSI(7)<35 + price<BB_lower*1.005
- SHORT: 4h_HMA bearish + RSI(7)>65 + price>BB_upper*0.995
- Also: RSI(7) cross above 50 in bullish 4h trend (momentum entry)
- Also: RSI(7) cross below 50 in bearish 4h trend (momentum entry)

Target: Sharpe>0.45, trades>=50 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.25 discrete (smaller for 15m frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi7_bb_4h_trend_session_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def get_hour_from_open_time(open_time_col):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = (open_time_col // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)  # Faster RSI for 15m
    rsi_14 = calculate_rsi(close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Session filter: 00-12 UTC only (London/NY overlap)
    utc_hours = get_hour_from_open_time(open_time)
    in_session = (utc_hours >= 0) & (utc_hours < 12)
    
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
    
    # Track RSI crosses for momentum entries
    prev_rsi_7 = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi_7 = rsi_7[i] if not np.isnan(rsi_7[i]) else prev_rsi_7
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi_7 = rsi_7[i] if not np.isnan(rsi_7[i]) else prev_rsi_7
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi_7 = rsi_7[i] if not np.isnan(rsi_7[i]) else prev_rsi_7
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi_7 = rsi_7[i] if not np.isnan(rsi_7[i]) else prev_rsi_7
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === BB POSITION ===
        bb_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        near_lower = close[i] < bb_lower[i] * 1.005  # Within 0.5% of lower band
        near_upper = close[i] > bb_upper[i] * 0.995  # Within 0.5% of upper band
        
        # === RSI CROSS DETECTION ===
        rsi_cross_above_50 = (prev_rsi_7 < 50.0 and rsi_7[i] >= 50.0) if prev_rsi_7 > 0 else False
        rsi_cross_below_50 = (prev_rsi_7 > 50.0 and rsi_7[i] <= 50.0) if prev_rsi_7 > 0 else False
        
        # === ENTRY LOGIC (LOOSE to guarantee trades) ===
        desired_signal = 0.0
        
        # Only trade during active session (00-12 UTC)
        if in_session[i]:
            # LONG entries (4h bullish bias)
            if hma_4h_bull:
                # Mean reversion: RSI(7) oversold + near BB lower
                if rsi_7[i] < 35.0 and near_lower:
                    desired_signal = SIZE_STRONG
                # Momentum: RSI(7) crosses above 50
                elif rsi_cross_above_50 and rsi_14[i] > 45.0:
                    desired_signal = SIZE_BASE
                # Simple RSI oversold in bullish trend
                elif rsi_7[i] < 30.0:
                    desired_signal = SIZE_BASE
            
            # SHORT entries (4h bearish bias)
            elif hma_4h_bear:
                # Mean reversion: RSI(7) overbought + near BB upper
                if rsi_7[i] > 65.0 and near_upper:
                    desired_signal = -SIZE_STRONG
                # Momentum: RSI(7) crosses below 50
                elif rsi_cross_below_50 and rsi_14[i] < 55.0:
                    desired_signal = -SIZE_BASE
                # Simple RSI overbought in bearish trend
                elif rsi_7[i] > 70.0:
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
        prev_rsi_7 = rsi_7[i] if not np.isnan(rsi_7[i]) else prev_rsi_7
    
    return signals