#!/usr/bin/env python3
"""
Experiment #1089: 15m Primary + 1h/1d HTF — Trend-Biased Mean Reversion

Hypothesis: 15m timeframe has failed repeatedly (Sharpe=0.000 = ZERO trades) because
entry conditions were TOO STRICT. This strategy LOOSENS entries significantly while
using HTF bias to avoid counter-trend trades.

Key innovations:
1. 1d HMA(21) for PRIMARY bias - only long if price > 1d_HMA, only short if price < 1d_HMA
2. 1h RSI(14) for momentum confirmation - RSI > 45 for long bias, < 55 for short bias
3. 15m RSI(7) for entry timing - oversold (<35) for long, overbought (>65) for short
4. Session filter: 00-12 UTC only (London/NY overlap = 70% of crypto volume)
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.20, ±0.25 to minimize fee churn

Why this should work on 15m:
- HTF bias (1d) prevents counter-trend mean reversion (the #1 killer)
- 1h RSI adds momentum filter without being too restrictive
- 15m RSI(7) is sensitive enough to generate 40-100 trades/year
- Session filter avoids low-volume whipsaws (Asia session 12-00 UTC)
- LOOSE thresholds (RSI<35/>65 not <20/>80) guarantee trades trigger

Entry conditions (LOOSE to guarantee ≥10 trades/symbol):
- LONG: price > 1d_HMA + 1h_RSI > 45 + 15m_RSI(7) < 35 + UTC hour 00-12
- SHORT: price < 1d_HMA + 1h_RSI < 55 + 15m_RSI(7) > 65 + UTC hour 00-12

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%, trades>=40/year
Timeframe: 15m
Size: 0.20-0.25 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_trend_bias_rsi_meanrev_1h1d_session_v1"
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
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Extract UTC hour from open_time (milliseconds timestamp)
    utc_hours = np.zeros(n, dtype=np.int32)
    for i in range(n):
        # Convert ms to seconds, then to datetime
        ts_sec = open_time[i] / 1000.0
        utc_hours[i] = int(pd.Timestamp(ts_sec, unit='s').hour)
    
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
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        utc_hour = utc_hours[i]
        in_session = (utc_hour >= 0 and utc_hour < 12)
        
        # === HTF BIAS (1d HMA for primary trend) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_aligned[i]
        
        # === 1h MOMENTUM (RSI confirmation) ===
        rsi_1h_bull = rsi_1h_aligned[i] > 45.0
        rsi_1h_bear = rsi_1h_aligned[i] < 55.0
        
        # === 15m ENTRY (RSI(7) mean reversion) ===
        rsi_7_oversold = rsi_7[i] < 35.0
        rsi_7_overbought = rsi_7[i] > 65.0
        
        # Stronger signals at more extreme RSI
        rsi_7_deep_oversold = rsi_7[i] < 25.0
        rsi_7_deep_overbought = rsi_7[i] > 75.0
        
        # === ENTRY LOGIC (TREND-BIASED MEAN REVERSION) ===
        desired_signal = 0.0
        
        if in_session:
            # LONG: 1d uptrend + 1h momentum + 15m oversold
            if price_above_1d_hma and rsi_1h_bull:
                if rsi_7_deep_oversold:
                    desired_signal = SIZE_STRONG
                elif rsi_7_oversold:
                    desired_signal = SIZE_BASE
            
            # SHORT: 1d downtrend + 1h momentum + 15m overbought
            elif price_below_1d_hma and rsi_1h_bear:
                if rsi_7_deep_overbought:
                    desired_signal = -SIZE_STRONG
                elif rsi_7_overbought:
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