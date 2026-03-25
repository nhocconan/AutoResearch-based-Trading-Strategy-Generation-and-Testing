#!/usr/bin/env python3
"""
Experiment #1341: 15m Primary + 1h/4h/1d HTF — Daily Pivot + 4h Trend + 15m RSI Pullback

Hypothesis: 15m has ZERO successful experiments because most try pure mean-reversion 
or pure trend-following. This combines BOTH with LOOSE entry conditions to guarantee trades:

1. Daily CPR (Central Pivot Range) from 1d HTF - key S/R levels where price reacts
2. 4h HMA(21) for major trend direction (only trade with HTF trend)
3. 15m RSI(7) for oversold/overbought pullback entries (LOOSE: <40 />60)
4. Session bonus 00-12 UTC (London/NY overlap = higher volume, but NOT required)
5. Discrete sizing 0.15-0.20 (smaller due to 15m frequency)
6. ATR 2.0x trailing stop

Entry logic (LOOSE to guarantee 40-100 trades/year):
- LONG: 4h HMA rising + price > daily pivot + 15m RSI(7) < 40 (oversold bounce)
- SHORT: 4h HMA falling + price < daily pivot + 15m RSI(7) > 60 (overbought fade)
- Session filter is BONUS not requirement (avoids 0 trades)

Why this should work on 15m:
- HTF trend filter reduces whipsaws (4h HMA = stable direction)
- Daily pivot = natural S/R where institutions react
- RSI(7) pullback = enters on retracements, not breakouts (better R:R)
- LOOSE thresholds guarantee trades (learned from 1102 failures)
- Small size (0.15-0.20) controls drawdown on higher frequency

Target: Sharpe>0.5, trades=40-100/year, DD>-35%, ALL symbols must trade
Timeframe: 15m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_daily_pivot_4h_trend_rsi_pullback_v1"
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
    
    return rsi

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

def calculate_daily_pivot(df_1d):
    """Calculate Daily CPR (Central Pivot Range) from daily data"""
    n = len(df_1d)
    
    pivot = np.full(n, np.nan, dtype=np.float64)
    bc = np.full(n, np.nan, dtype=np.float64)
    tc = np.full(n, np.nan, dtype=np.float64)
    r1 = np.full(n, np.nan, dtype=np.float64)
    s1 = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(1, n):
        high_prev = df_1d['high'].iloc[i-1]
        low_prev = df_1d['low'].iloc[i-1]
        close_prev = df_1d['close'].iloc[i-1]
        
        pivot[i] = (high_prev + low_prev + close_prev) / 3.0
        bc[i] = (high_prev + low_prev) / 2.0
        tc[i] = (pivot[i] - bc[i]) + pivot[i]
        r1[i] = 2.0 * pivot[i] - low_prev
        s1[i] = 2.0 * pivot[i] - high_prev
    
    return pivot, bc, tc, r1, s1

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate daily pivot levels and align
    pivot_1d, bc_1d, tc_1d, r1_1d, s1_1d = calculate_daily_pivot(df_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    bc_aligned = align_htf_to_ltf(prices, df_1d, bc_1d)
    tc_aligned = align_htf_to_ltf(prices, df_1d, tc_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
    
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(pivot_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION BONUS (00-12 UTC) - NOT REQUIRED, just bonus ===
        hour_utc = (open_time[i] // 1000 // 3600) % 24
        in_session = 0 <= hour_utc < 12
        
        # === 4h TREND DIRECTION (LOOSE - use 8 bars for slope) ===
        hma_4h_slope = 0.0
        if i >= 8 and not np.isnan(hma_4h_aligned[i-8]):
            hma_4h_slope = hma_4h_aligned[i] - hma_4h_aligned[i-8]
        
        trend_bullish = hma_4h_slope > 0
        trend_bearish = hma_4h_slope < 0
        
        # === DAILY PIVOT POSITION ===
        price_above_pivot = close[i] > pivot_aligned[i]
        price_below_pivot = close[i] < pivot_aligned[i]
        
        # Check if near support/resistance (within 1.5% of pivot levels)
        near_s1 = False
        near_r1 = False
        if not np.isnan(s1_aligned[i]) and s1_aligned[i] > 0:
            near_s1 = abs(close[i] - s1_aligned[i]) / close[i] < 0.015
        if not np.isnan(r1_aligned[i]) and r1_aligned[i] > 0:
            near_r1 = abs(close[i] - r1_aligned[i]) / close[i] < 0.015
        
        # === 15m RSI PULLBACK (LOOSE thresholds) ===
        rsi = rsi_7[i]
        oversold = rsi < 40  # LOOSE: was 35
        overbought = rsi > 60  # LOOSE: was 65
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        # Only 2 confluence required: HTF trend + RSI extreme
        # Pivot position and session are bonuses
        desired_signal = 0.0
        
        # LONG: 4h trend up + RSI oversold (pivot above is bonus)
        if trend_bullish and oversold:
            confluence = 2  # trend + RSI
            if price_above_pivot:
                confluence += 1
            if in_session:
                confluence += 1
            if near_s1:
                confluence += 1
            
            if confluence >= 2:  # LOOSE: only 2 required
                if confluence >= 4:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: 4h trend down + RSI overbought (pivot below is bonus)
        elif trend_bearish and overbought:
            confluence = 2  # trend + RSI
            if price_below_pivot:
                confluence += 1
            if in_session:
                confluence += 1
            if near_r1:
                confluence += 1
            
            if confluence >= 2:  # LOOSE: only 2 required
                if confluence >= 4:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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