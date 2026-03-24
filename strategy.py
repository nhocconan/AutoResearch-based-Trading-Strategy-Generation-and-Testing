#!/usr/bin/env python3
"""
Experiment #972: 12h Primary + 1d HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: 12h timeframe with Donchian breakout entries filtered by 1d HMA trend
will generate consistent trades with good risk/reward in mixed 2021-2025 markets.

Key innovations:
1. Donchian(20) breakout: price breaks 20-bar high/low for momentum entry
2. 1d HMA(21) for trend bias — only trade breakouts in trend direction
3. RSI(14) filter: avoid overbought longs (>70) and oversold shorts (<30)
4. ATR(14) 2.5x trailing stop for risk management
5. LOOSE entry conditions to guarantee 30+ trades/train, 5+/test

Why this should work:
- Donchian breakouts catch sustained moves (proven on SOL in prior exps)
- 12h TF = 20-50 trades/year target (low fee drag)
- 1d HMA filter prevents counter-trend breakout failures
- RSI filter avoids chasing exhausted moves
- Simpler than CHOP/CRSI regime switching = more trades triggered

Entry conditions (LOOSE to guarantee trades):
- LONG = price>SMA200 + 1d HMA bull + Donchian breakout + RSI<70
- SHORT = price<SMA200 + 1d HMA bear + Donchian breakdown + RSI>30
- Relaxed RSI thresholds (30/70 vs 20/80) for more trade frequency

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_rsi_1d_v1"
timeframe = "12h"
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    
    return sma

def calculate_donchian(high, low, period=20):
    """Donchian Channel — returns (upper, lower)"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

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
    sma_200 = calculate_sma(close, period=200)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
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
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND FILTER (SMA200) ===
        primary_bull = close[i] > sma_200[i]
        primary_bear = close[i] < sma_200[i]
        
        # === DONCHIAN BREAKOUT DETECTION ===
        donchian_breakout_long = False
        donchian_breakout_short = False
        
        # Check if price breaks Donchian upper/lower
        if close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False:
            donchian_breakout_long = True
        if close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False:
            donchian_breakout_short = True
        
        # === RSI FILTER (LOOSE THRESHOLDS FOR MORE TRADES) ===
        rsi_ok_long = rsi_14[i] < 70  # Not overbought
        rsi_ok_short = rsi_14[i] > 30  # Not oversold
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG entries — multiple conditions, any can trigger
        if htf_1d_bull and primary_bull:
            if donchian_breakout_long and rsi_ok_long:
                # Strong breakout with trend
                desired_signal = SIZE_STRONG
            elif close[i] > donchian_upper[i] * 0.99 and rsi_14[i] < 60:
                # Near breakout with moderate RSI
                desired_signal = SIZE_BASE
            elif rsi_14[i] < 40 and close[i] > sma_200[i] * 1.01:
                # Pullback entry in uptrend
                desired_signal = SIZE_BASE
        
        # SHORT entries — multiple conditions, any can trigger
        elif htf_1d_bear and primary_bear:
            if donchian_breakout_short and rsi_ok_short:
                # Strong breakdown with trend
                desired_signal = -SIZE_STRONG
            elif close[i] < donchian_lower[i] * 1.01 and rsi_14[i] > 40:
                # Near breakdown with moderate RSI
                desired_signal = -SIZE_BASE
            elif rsi_14[i] > 60 and close[i] < sma_200[i] * 0.99:
                # Pullback entry in downtrend
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