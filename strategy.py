#!/usr/bin/env python3
"""
Experiment #654: 4h Primary + 12h/1d HTF — HMA Trend + RSI Pullback + Donchian Breakout

Hypothesis: 4h timeframe with 12h trend bias provides optimal balance between signal 
quality and trade frequency. Proven pattern from best strategies: HMA for trend, RSI 
for pullback entries, Donchian for breakout confirmation. 

Key innovations:
1. LOOSE entry conditions — RSI 35/65 (not 30/70) to ensure adequate trades
2. OR logic for entries — multiple independent entry triggers (not all must align)
3. 12h HMA as soft bias — only filters extreme counter-trend, doesn't block all trades
4. Donchian(20) breakout confirmation — catches momentum moves
5. Hold logic maintains positions through minor pullbacks (critical for trade count)
6. ATR trailing stop at 2.5x — protects capital without premature exits

Why this should beat Sharpe=0.612:
- 4h timeframe = proven winner (current best is 4h-based)
- Looser RSI thresholds = more trades (addresses #1 failure: 0 trades)
- Multiple entry triggers = higher hit rate across different market conditions
- 12h HTF = trend filter without being too restrictive
- Conservative sizing (0.28) survives crashes while capturing gains

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_donchian_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — smoother and more responsive than EMA."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """RSI with proper min_periods."""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel — highest high and lowest low over period."""
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (hh + ll) / 2
    return hh, ll, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h indicators (primary timeframe)
    hma_4h_16 = calculate_hma(close, period=16)
    hma_4h_48 = calculate_hma(close, period=48)
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    donch_hh, donch_ll, donch_mid = calculate_donchian(high, low, period=20)
    
    # Calculate and align HTF indicators (12h)
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.28
    SIZE_SHORT = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            continue
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if np.isnan(donch_hh[i]) or np.isnan(donch_ll[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or atr_4h[i] <= 1e-10:
            continue
        
        # === HTF TREND BIAS (12h HMA) — SOFT FILTER ===
        htf_12h_bullish = close[i] > hma_12h_aligned[i]
        htf_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # === 4h TREND (HMA crossover) ===
        hma_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_bearish = hma_4h_16[i] < hma_4h_48[i]
        
        # === RSI SIGNALS (LOOSE thresholds for trade frequency) ===
        rsi_oversold = rsi_4h[i] < 35.0
        rsi_overbought = rsi_4h[i] > 65.0
        rsi_neutral = 35.0 <= rsi_4h[i] <= 65.0
        
        # === DONCHIAN BREAKOUT ===
        donch_breakout_long = close[i] > donch_hh[i-1] if i > 0 else False
        donch_breakout_short = close[i] < donch_ll[i-1] if i > 0 else False
        
        desired_signal = 0.0
        
        # === LONG ENTRY TRIGGERS (OR logic — any one can trigger) ===
        long_trigger_1 = hma_bullish and rsi_oversold and htf_12h_bullish
        long_trigger_2 = hma_bullish and donch_breakout_long
        long_trigger_3 = htf_12h_bullish and rsi_oversold and rsi_4h[i] > rsi_4h[i-1] if i > 0 else False
        long_trigger_4 = hma_bullish and htf_12h_bullish and rsi_neutral
        
        # === SHORT ENTRY TRIGGERS (OR logic — any one can trigger) ===
        short_trigger_1 = hma_bearish and rsi_overbought and htf_12h_bearish
        short_trigger_2 = hma_bearish and donch_breakout_short
        short_trigger_3 = htf_12h_bearish and rsi_overbought and rsi_4h[i] < rsi_4h[i-1] if i > 0 else False
        short_trigger_4 = hma_bearish and htf_12h_bearish and rsi_neutral
        
        # Entry logic — OR triggers
        if long_trigger_1 or long_trigger_2 or long_trigger_3 or long_trigger_4:
            desired_signal = SIZE_LONG
        elif short_trigger_1 or short_trigger_2 or short_trigger_3 or short_trigger_4:
            desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — CRITICAL for trade frequency ===
        # Maintain position if trend unchanged (prevents premature exits)
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HMA still bullish OR RSI not extremely overbought
                if hma_bullish and rsi_4h[i] < 75.0:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HMA still bearish OR RSI not extremely oversold
                if hma_bearish and rsi_4h[i] > 25.0:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            # If same side, update trailing stop levels
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals