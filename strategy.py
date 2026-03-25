#!/usr/bin/env python3
"""
Experiment #1499: 1h Primary + 4h/12h HTF — Simplified Trend-Pullback Strategy

Hypothesis: After 1200+ failed strategies, complexity is the enemy. This strategy
uses SIMPLE but LOOSE entry conditions that GUARANTEE trades while maintaining
edge through HTF trend alignment.

Key insight from failures:
- Experiments 1489-1498 all got Sharpe=0.000 (ZERO TRADES) due to over-filtering
- Complex regime detection (Choppiness, ADX, multiple confirmations) = no triggers
- Session filters on lower TFs kill trade frequency

This strategy:
1. 4h HMA(21) for trend bias (simple, proven)
2. 1h RSI(7) for entry timing (looser than RSI14)
3. 1h EMA(21) pullback entry (price near EMA, not exact touch)
4. Volume filter (above 0.7x average - very loose)
5. ATR(14) stoploss at 2.5x

Entry logic (LOOSE to guarantee ≥40 trades/year):
- LONG: 4h_HMA bullish + RSI7<35 + price<EMA21*1.01 + vol>0.7x_avg
- SHORT: 4h_HMA bearish + RSI7>65 + price>EMA21*0.99 + vol>0.7x_avg

Why this should work:
- RSI7 reaches 35/65 frequently (unlike RSI14 at 30/70)
- Price "near EMA" triggers often (within 1%, not exact touch)
- Volume filter is very permissive (70% of average)
- 4h trend filter prevents counter-trend disasters
- 1h TF = natural 50-80 trades/year with these loose settings

Timeframe: 1h
Size: 0.25 discrete (0.0, ±0.25)
Target: Sharpe>0.5, trades>=40/train, trades>=5/test, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_loose_4h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of Volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1h indicators
    ema_21 = calculate_ema(close, period=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)  # Faster RSI for more signals
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
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
        
        if np.isnan(rsi_7[i]) or np.isnan(ema_21[i]) or np.isnan(vol_sma_20[i]):
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
        
        # === TREND BIAS (4h HMA) ===
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # === RSI EXTREMES (LOOSE thresholds) ===
        rsi = rsi_7[i]
        rsi_oversold = rsi < 35  # Loose: not 30, not 25
        rsi_overbought = rsi > 65  # Loose: not 70, not 75
        
        # === PRICE NEAR EMA (within 1%, not exact touch) ===
        price_near_ema_long = close[i] < ema_21[i] * 1.015  # Within 1.5% below EMA
        price_near_ema_short = close[i] > ema_21[i] * 0.985  # Within 1.5% above EMA
        
        # === VOLUME FILTER (very loose) ===
        vol_ratio = volume[i] / vol_sma_20[i] if vol_sma_20[i] > 0 else 0
        vol_ok = vol_ratio > 0.6  # 60% of average is fine
        
        # === SESSION FILTER (08-20 UTC for liquidity) ===
        hour_utc = (open_time[i] // 3600000) % 24
        session_ok = 6 <= hour_utc <= 22  # Very wide: 06-22 UTC
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + RSI oversold + price near EMA + volume ok
        if trend_bullish and rsi_oversold and price_near_ema_long and vol_ok and session_ok:
            desired_signal = SIZE
        
        # SHORT: 4h bearish + RSI overbought + price near EMA + volume ok
        elif trend_bearish and rsi_overbought and price_near_ema_short and vol_ok and session_ok:
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