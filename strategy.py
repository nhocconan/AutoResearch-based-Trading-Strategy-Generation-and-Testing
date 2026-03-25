#!/usr/bin/env python3
"""
Experiment #1253: 5m Primary + 15m/4h HTF — Session-Filtered Trend Pullback

Hypothesis: 5m has ZERO prior experiments. Key insight: 5m needs EXTREME selectivity
to avoid fee drag from too many trades. This strategy uses:

1. 4h HMA(21) for PRIMARY trend direction (loaded ONCE before loop)
2. 15m RSI(14) for momentum confirmation (loaded ONCE before loop)
3. 5m EMA(21) pullback entries in HTF trend direction
4. Session filter: ONLY trade 08-20 UTC (high liquidity, avoid Asia chop)
5. ATR(14) 2.5x trailing stop for risk management

Why this should work on 5m:
- 4h trend filter = only trade in established direction (no counter-trend)
- Session filter = avoids low-liquidity hours where 5m gets whipsawed
- EMA pullback = enters on retracements, not breakouts (better risk/reward)
- Small size (0.15-0.20) = accounts for higher trade frequency
- Target: 50-120 trades/year (fee-friendly for 5m)

Entry logic:
- LONG: 4h_HMA bullish + 15m_RSI > 50 + 5m price pulls back to EMA(21)
- SHORT: 4h_HMA bearish + 15m_RSI < 50 + 5m price rallies to EMA(21)
- Session: only 08-20 UTC (12-hour window = ~50% of bars eligible)

Timeframe: 5m
Size: 0.15-0.20 discrete (smaller due to more trades)
Target: Sharpe>0.5, trades>=50 train, trades>=5 test, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_hma_pullback_15m4h_v1"
timeframe = "5m"
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

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def is_session_active(open_time_unix_ms, start_hour=8, end_hour=20):
    """Check if timestamp is within trading session (UTC)"""
    # Convert milliseconds to datetime
    dt = pd.to_datetime(open_time_unix_ms, unit='ms', utc=True)
    hour = dt.hour
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
    atr_14 = calculate_atr(high, low, close, period=14)
    ema_21 = calculate_ema(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(ema_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_15m_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC only) ===
        if not is_session_active(open_time[i], start_hour=8, end_hour=20):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (4h HMA) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === MOMENTUM CONFIRMATION (15m RSI) ===
        rsi_15m = rsi_15m_aligned[i]
        rsi_bullish = rsi_15m > 50.0
        rsi_bearish = rsi_15m < 50.0
        
        # === ENTRY LOGIC (Pullback to EMA in trend direction) ===
        desired_signal = 0.0
        
        # Price distance from EMA (for pullback detection)
        ema_distance = (close[i] - ema_21[i]) / ema_21[i] * 100.0 if ema_21[i] > 0 else 0.0
        
        # LONG: 4h bullish + 15m RSI bullish + price near/pulling back to EMA
        if price_above_4h and rsi_bullish:
            # Pullback: price within 0.5% of EMA or slightly below
            if ema_distance <= 0.5 and ema_distance >= -1.0:
                desired_signal = SIZE_BASE
                # Strong long: price actually touching EMA from above
                if ema_distance >= -0.3 and ema_distance <= 0.3:
                    desired_signal = SIZE_STRONG
        
        # SHORT: 4h bearish + 15m RSI bearish + price near/rallying to EMA
        elif price_below_4h and rsi_bearish:
            # Rally: price within 0.5% of EMA or slightly above
            if ema_distance >= -0.5 and ema_distance <= 1.0:
                desired_signal = -SIZE_BASE
                # Strong short: price actually touching EMA from below
                if ema_distance >= -0.3 and ema_distance <= 0.3:
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