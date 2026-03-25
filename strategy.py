#!/usr/bin/env python3
"""
Experiment #1187: 6h Primary + 1d HTF — Keltner Channel Mean Reversion + HMA Trend

Hypothesis: After 960+ failed experiments, the 6h timeframe is underexplored. Most failed
6h strategies used complex regime filters or weekly pivots that generated 0 trades.

This strategy uses Keltner Channels (EMA + ATR bands) for mean reversion entries:
- Keltner works well in ranging/bear markets (2025 test period)
- 1d HMA(21) provides simple trend bias without over-filtering
- Entry when price touches Keltner band + RSI extreme in trend direction
- Exit at Keltner middle (EMA) or 2.5x ATR stoploss

Why this should work on 6h:
- 6h = natural 30-60 trades/year (fee-friendly, not too many)
- Keltner mean reversion captures multi-day swings in crypto
- 1d trend filter avoids counter-trend trades in strong trends
- Different from all failed 6h strategies (no weekly pivots, no Donchian, no Fisher)

Entry logic:
- LONG: price > 1d_HMA + price touches Keltner lower band + RSI(14) < 40
- SHORT: price < 1d_HMA + price touches Keltner upper band + RSI(14) > 60
- Exit: price crosses Keltner middle (EMA20) or stoploss hit

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_keltner_meanreversion_hma_trend_1d_v1"
timeframe = "6h"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    ema_20 = calculate_ema(close, period=20)
    
    # Keltner Channels: EMA(20) +/- 2.0*ATR(14)
    keltner_mult = 2.0
    keltner_upper = ema_20 + keltner_mult * atr_14
    keltner_lower = ema_20 - keltner_mult * atr_14
    keltner_middle = ema_20
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(ema_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (Daily HMA) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === KELTNER CHANNEL POSITION ===
        price_near_lower = low[i] <= keltner_lower[i] * 1.002  # touched or crossed lower band
        price_near_upper = high[i] >= keltner_upper[i] * 0.998  # touched or crossed upper band
        price_at_middle = abs(close[i] - keltner_middle[i]) < keltner_middle[i] * 0.005  # near EMA
        
        # === ENTRY LOGIC (Mean Reversion in Trend Direction) ===
        desired_signal = 0.0
        rsi = rsi_14[i]
        
        # LONG: Price above 1d HMA (uptrend) + price touches Keltner lower + RSI oversold
        if price_above_1d:
            if price_near_lower and rsi < 40.0:
                desired_signal = SIZE_BASE
            elif price_near_lower and rsi < 30.0:
                desired_signal = SIZE_STRONG  # Stronger oversold
        
        # SHORT: Price below 1d HMA (downtrend) + price touches Keltner upper + RSI overbought
        elif price_below_1d:
            if price_near_upper and rsi > 60.0:
                desired_signal = -SIZE_BASE
            elif price_near_upper and rsi > 70.0:
                desired_signal = -SIZE_STRONG  # Stronger overbought
        
        # === EXIT LOGIC (Mean Reversion to Middle) ===
        if in_position:
            if position_side > 0 and price_at_middle:
                # Long position: exit when price returns to Keltner middle
                desired_signal = 0.0
            elif position_side < 0 and price_at_middle:
                # Short position: exit when price returns to Keltner middle
                desired_signal = 0.0
        
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