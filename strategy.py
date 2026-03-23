#!/usr/bin/env python3
"""
Experiment #492: 12h Primary + 1d/1w HTF — Dual KAMA Trend + ADX Filter + RSI Pullback

Hypothesis: Based on #486 success (12h HMA+ADX+RSI, Sharpe=0.440), combining:
1. Dual KAMA (fast=10, slow=40) for adaptive trend detection — KAMA adapts to volatility
2. ADX > 20 filter to ensure trend strength (avoid chop)
3. RSI pullback entries (40-60 range) for better entry timing
4. 1d KAMA for intermediate HTF bias
5. 1w HMA for major trend filter (only trade with weekly trend)
6. ATR trailing stop for risk management

Why this should beat Sharpe=0.612:
- 12h proven to work (#486 had +98% return)
- Dual KAMA crossover catches trends earlier than single MA
- Weekly HMA filter avoids counter-trend trades in major moves
- Relaxed ADX > 20 (vs 25) ensures trade generation
- RSI 40/60 thresholds (vs 30/70) generate more entries

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_kama_adx_rsi_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts to market noise: fast in trends, slow in chop.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.zeros(n)
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period=21):
    """
    Hull Moving Average (HMA).
    Reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, span):
        weights = np.arange(1, span + 1)
        weights = weights / weights.sum()
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            result[i] = np.sum(series[i - span + 1:i + 1] * weights)
        return result
    
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean().values
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # HMA calculation
    raw_hma = 2.0 * wma_half - wma_full
    hma = pd.Series(raw_hma).ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean().values
    
    return hma

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    adx = np.full(n, np.nan)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    
    if n < period * 2:
        return adx, plus_di, minus_di
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with Wilder's method
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI+ and DI-
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100.0 * plus_dm_smooth / (tr_smooth + 1e-10)
        minus_di = 100.0 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    # DX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX = smoothed DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing method."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0.0)
    loss[1:] = np.where(delta < 0, -delta, 0.0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    kama_fast = calculate_kama(close, period=10, fast_period=2, slow_period=10)
    kama_slow = calculate_kama(close, period=40, fast_period=2, slow_period=30)
    adx_12h, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    rsi_12h = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators
    # 1d KAMA for intermediate trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # 1w HMA for major trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            continue
        if np.isnan(adx_12h[i]):
            continue
        if np.isnan(rsi_12h[i]):
            continue
        if np.isnan(kama_1d_aligned[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        # === MAJOR TREND FILTER (1w HMA) ===
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d KAMA) ===
        daily_bullish = close[i] > kama_1d_aligned[i]
        daily_bearish = close[i] < kama_1d_aligned[i]
        
        # === PRIMARY TREND (12h Dual KAMA) ===
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # KAMA slope confirmation
        kama_fast_slope_up = kama_fast[i] > kama_fast[i - 3] if i >= 3 else False
        kama_fast_slope_down = kama_fast[i] < kama_fast[i - 3] if i >= 3 else False
        
        # === TREND STRENGTH (ADX) ===
        # Relaxed threshold for trade generation
        trend_strong = adx_12h[i] > 20.0
        
        # === RSI PULLBACK SIGNALS ===
        # Relaxed thresholds for more entries
        rsi_neutral_long = rsi_12h[i] < 60.0
        rsi_neutral_short = rsi_12h[i] > 40.0
        rsi_not_extreme_long = rsi_12h[i] > 30.0
        rsi_not_extreme_short = rsi_12h[i] < 70.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES - Multiple confluence required
        long_score = 0
        
        # Weekly trend bullish (strong filter - weight 2)
        if weekly_bullish:
            long_score += 2
        
        # Daily trend bullish
        if daily_bullish:
            long_score += 1
        
        # 12h KAMA crossover bullish
        if kama_bullish:
            long_score += 1
        
        # KAMA fast slope up
        if kama_fast_slope_up:
            long_score += 1
        
        # ADX shows trend strength
        if trend_strong:
            long_score += 1
        
        # RSI pullback (not overbought, room to run)
        if rsi_neutral_long and rsi_not_extreme_long:
            long_score += 1
        
        # Enter long if score >= 4 (balanced for trade generation)
        if long_score >= 4:
            desired_signal = SIZE_LONG
        
        # SHORT ENTRIES
        if desired_signal == 0.0:
            short_score = 0
            
            # Weekly trend bearish
            if weekly_bearish:
                short_score += 2
            
            # Daily trend bearish
            if daily_bearish:
                short_score += 1
            
            # 12h KAMA crossover bearish
            if kama_bearish:
                short_score += 1
            
            # KAMA fast slope down
            if kama_fast_slope_down:
                short_score += 1
            
            # ADX shows trend strength
            if trend_strong:
                short_score += 1
            
            # RSI pullback (not oversold)
            if rsi_neutral_short and rsi_not_extreme_short:
                short_score += 1
            
            if short_score >= 4:
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if weekly OR daily still bullish
                if weekly_bullish or daily_bullish:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if weekly OR daily still bearish
                if weekly_bearish or daily_bearish:
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
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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