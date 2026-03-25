#!/usr/bin/env python3
"""
Experiment #1604: 12h Primary + 1d/1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Previous 12h strategies failed due to OVER-FILTERING. Too many confluence
conditions = 0 trades. This strategy SIMPLIFIES entry logic while keeping MTF structure.

Key changes vs failed attempts:
1. REMOVED: Fisher Transform (too complex, failed in #1600)
2. REMOVED: Choppiness Index regime (unreliable, failed in #1594)
3. REMOVED: Volume confirmation (filters too many valid trades)
4. SIMPLIFIED: Entry = HMA trend + RSI pullback ONLY (proven pattern)
5. LOOSENED: RSI thresholds 35/65 (not 30/70) to guarantee trades
6. ADDED: Asymmetric sizing (0.30 long, 0.25 short) based on crypto bias
7. ADDED: 1w HMA as ultimate trend filter (prevents major counter-trend)

Why this should work when others failed:
- Fewer filters = more trades (target 40-60 trades/year on 12h)
- HMA trend + RSI pullback is proven in mtf_6h_triple_hma (Sharpe=0.575)
- 12h TF = lower fee drag than 6h, same edge
- Asymmetric sizing accounts for crypto long bias

Entry logic (SIMPLE to guarantee trades):
- LONG: 12h_HMA bullish + 1d_HMA bullish + 1w_HMA bullish + RSI(14) < 55 (pullback)
- SHORT: 12h_HMA bearish + 1d_HMA bearish + RSI(14) > 45 (rally into weakness)
- Exit: RSI crosses 50 opposite direction OR stoploss hit

Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 12h
Size: 0.25-0.30 discrete (asymmetric: long=0.30, short=0.25)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_pullback_1d1w_asym_v2"
timeframe = "12h"
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / tr_smooth
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / tr_smooth
    
    dx = np.zeros(n, dtype=np.float64)
    mask = (plus_di + minus_di) > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 12h indicators
    hma_12h = calculate_hma(close, period=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30  # Asymmetric: larger long position (crypto bias)
    SIZE_SHORT = 0.25  # Smaller short position
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track RSI for exit signals
    prev_rsi = np.nan
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_12h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (Multi-timeframe HMA) ===
        price_above_12h = close[i] > hma_12h[i]
        price_below_12h = close[i] < hma_12h[i]
        
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === RSI PULLBACK SIGNALS (LOOSE thresholds for trades) ===
        rsi_val = rsi_14[i]
        rsi_prev = prev_rsi if not np.isnan(prev_rsi) else rsi_val
        
        # RSI pullback in uptrend (RSI drops but still > 40)
        rsi_pullback_long = rsi_val < 55 and rsi_val > 35
        rsi_pullback_short = rsi_val > 45 and rsi_val < 65
        
        # RSI extreme reversal
        rsi_oversold = rsi_val < 35
        rsi_overbought = rsi_val > 65
        
        # RSI crossover signals
        rsi_cross_up = rsi_val > 50 and rsi_prev <= 50
        rsi_cross_down = rsi_val < 50 and rsi_prev >= 50
        
        # === ADX TREND STRENGTH ===
        adx_val = adx_14[i] if not np.isnan(adx_14[i]) else 0
        is_trending = adx_val > 20
        is_strong_trend = adx_val > 25
        
        # === ENTRY LOGIC (SIMPLE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG entries (asymmetric: easier to enter long)
        # Condition 1: Strong trend long (all HTF bullish + RSI pullback)
        if price_above_1w and price_above_1d and price_above_12h:
            if rsi_pullback_long or rsi_oversold:
                desired_signal = SIZE_LONG
        
        # Condition 2: Moderate trend long (12h + 1d bullish, 1w neutral/bullish)
        elif price_above_1d and price_above_12h and rsi_pullback_long:
            desired_signal = SIZE_LONG * 0.7  # Reduced size
        
        # SHORT entries (stricter conditions)
        # Condition 1: Strong trend short (all HTF bearish + RSI rally)
        elif price_below_1w and price_below_1d and price_below_12h:
            if rsi_pullback_short or rsi_overbought:
                desired_signal = -SIZE_SHORT
        
        # Condition 2: Moderate trend short (12h + 1d bearish)
        elif price_below_1d and price_below_12h and rsi_pullback_short:
            desired_signal = -SIZE_SHORT * 0.7  # Reduced size
        
        # === EXIT SIGNALS (RSI cross or stoploss) ===
        exit_signal = False
        
        # Exit long when RSI crosses above 60 (overbought in uptrend)
        if in_position and position_side > 0:
            if rsi_val > 60 or rsi_cross_down:
                exit_signal = True
        
        # Exit short when RSI crosses below 40 (oversold in downtrend)
        if in_position and position_side < 0:
            if rsi_val < 40 or rsi_cross_up:
                exit_signal = True
        
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
            exit_signal = True
        
        # === APPLY EXIT SIGNAL ===
        if exit_signal and in_position:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_LONG * 0.9:
            final_signal = SIZE_LONG
        elif desired_signal <= -SIZE_SHORT * 0.9:
            final_signal = -SIZE_SHORT
        elif desired_signal >= SIZE_LONG * 0.5:
            final_signal = SIZE_LONG * 0.7
        elif desired_signal <= -SIZE_SHORT * 0.5:
            final_signal = -SIZE_SHORT * 0.7
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
        prev_rsi = rsi_val
    
    return signals