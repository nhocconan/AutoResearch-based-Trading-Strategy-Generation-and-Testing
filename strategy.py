#!/usr/bin/env python3
"""
Experiment #1632: 12h Primary + 1d HTF — Simple Trend Pullback Strategy

Hypothesis: After 1327 failed experiments, complexity is the enemy. The best performer
(mtf_hma_rsi_zscore_v1 Sharpe=5.4) used simple HMA trend + RSI pullback logic.
Recent failures show regime-switching and complex filters cause 0 trades.

Key design choices based on failure analysis:
1. SIMPLE logic: 1d HMA trend bias + 12h RSI pullback entries only
2. LOOSE thresholds: RSI 35/65 (not 30/70), ADX > 18 (not > 25)
3. NO regime switching (Choppiness, Fisher regimes all failed recently)
4. NO volume filters (causes 0 trades per #1607, #1612)
5. Single HTF only (1d HMA, not 1d+1w which reduces trades per #1624)
6. Discrete signal sizes: 0.25 base, 0.30 strong conviction
7. 3.0x ATR trailing stoploss via signal→0

Why 12h works:
- 20-50 trades/year target (vs 100+ on 1h which causes fee drag)
- Catches major moves without whipsaw noise
- 1d HTF provides clear trend bias for bear/range markets (2022-2025)

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3 test):
- LONG: 1d HMA bullish + RSI(14) 35-50 pullback + ADX > 18
- SHORT: 1d HMA bearish + RSI(14) 50-65 pullback + ADX > 18
- Exit: RSI crosses 50 against position OR stoploss hit

Target: Sharpe>0.6, trades≥30 train, trades≥3 test, DD>-35%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_pullback_1d_simple_v1"
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
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    mask = atr > 1e-10
    plus_di[mask] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[mask] / atr[mask]
    minus_di[mask] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[mask] / atr[mask]
    
    dx = np.zeros(n, dtype=np.float64)
    di_sum = plus_di + minus_di
    mask2 = di_sum > 1e-10
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

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
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
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
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
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
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === TREND STRENGTH (ADX) ===
        adx_val = adx_14[i]
        trend_strong = adx_val > 18  # LOOSE threshold for trades
        
        # === RSI PULLBACK (LOOSE thresholds) ===
        rsi_val = rsi_14[i]
        rsi_prev = rsi_14[i-1] if i > 0 and not np.isnan(rsi_14[i-1]) else rsi_val
        
        # Pullback zones (LOOSE to guarantee trades)
        rsi_pullback_long = 35 <= rsi_val <= 55  # Pullback in uptrend
        rsi_pullback_short = 45 <= rsi_val <= 65  # Pullback in downtrend
        
        # RSI momentum confirmation
        rsi_rising = rsi_val > rsi_prev
        rsi_falling = rsi_val < rsi_prev
        
        # === ENTRY LOGIC (SIMPLE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + trend strong + RSI pullback + RSI turning up
        if price_above_1d and trend_strong and rsi_pullback_long and rsi_rising:
            # Strong signal if RSI was very oversold
            if rsi_val <= 40:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: 1d bearish + trend strong + RSI pullback + RSI turning down
        elif price_below_1d and trend_strong and rsi_pullback_short and rsi_falling:
            # Strong signal if RSI was very overbought
            if rsi_val >= 60:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (3.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        # === EXIT SIGNALS (RSI cross 50 against position) ===
        exit_signal = False
        if in_position and position_side > 0 and rsi_val > 65:
            exit_signal = True  # Overbought exit
        if in_position and position_side < 0 and rsi_val < 35:
            exit_signal = True  # Oversold exit
        
        if stoploss_triggered or exit_signal:
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
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
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