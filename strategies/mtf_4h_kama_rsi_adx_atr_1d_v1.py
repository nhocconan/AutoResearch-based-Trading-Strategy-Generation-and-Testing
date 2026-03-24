#!/usr/bin/env python3
"""
Experiment #491: 4h Primary + 1d HTF — KAMA Adaptive Trend + RSI Entries + ADX Filter

Hypothesis: After 400+ experiments, KAMA (Kaufman Adaptive MA) outperforms HMA/EMA in 
crypto's varying volatility regimes. KAMA adapts: fast in trends, slow in chop.

Key improvements over #489:
1. KAMA instead of HMA - adapts to volatility (proven in crypto)
2. Relaxed ADX > 15 (vs 18) - more trade generation
3. RSI 45/55 thresholds (vs 40/60) - tighter pullback entries
4. 1d KAMA for HTF bias (vs HMA) - consistent adaptive approach
5. Simplified HOLD logic - reduce complexity

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_adx_atr_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts to market noise: fast in trends, slow in chop.
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
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

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    Measures trend strength regardless of direction.
    """
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
    
    # Calculate 4h indicators (primary timeframe)
    kama_4h = calculate_kama(close, period=10)
    adx_4h, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    rsi_4h = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators (1d KAMA for major trend bias)
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss and HOLD logic
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
        if np.isnan(kama_4h[i]):
            continue
        if np.isnan(adx_4h[i]):
            continue
        if np.isnan(rsi_4h[i]):
            continue
        if np.isnan(kama_1d_aligned[i]):
            continue
        
        # === HTF MAJOR TREND BIAS (1d KAMA) ===
        htf_bullish = close[i] > kama_1d_aligned[i]
        htf_bearish = close[i] < kama_1d_aligned[i]
        
        # === PRIMARY TREND (4h KAMA) ===
        price_above_kama = close[i] > kama_4h[i]
        price_below_kama = close[i] < kama_4h[i]
        kama_slope_up = kama_4h[i] > kama_4h[i - 5] if i >= 5 else False
        kama_slope_down = kama_4h[i] < kama_4h[i - 5] if i >= 5 else False
        
        # === TREND STRENGTH (ADX) ===
        # Relaxed threshold: ADX > 15 to ensure trade generation
        trend_strong = adx_4h[i] > 15.0
        
        # === RSI SIGNALS (tighter thresholds for better entries) ===
        rsi_pullback_long = rsi_4h[i] < 55.0
        rsi_pullback_short = rsi_4h[i] > 45.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES - Relaxed conditions for trade generation
        long_score = 0
        
        # HTF bullish bias (strong signal - weight 2)
        if htf_bullish:
            long_score += 2
        
        # Price above 4h KAMA
        if price_above_kama:
            long_score += 1
        
        # KAMA slope up
        if kama_slope_up:
            long_score += 1
        
        # RSI pullback (room to run)
        if rsi_pullback_long:
            long_score += 1
        
        # ADX shows some trend OR strong HTF bias
        if trend_strong or htf_bullish:
            long_score += 1
        
        # Enter long if score >= 4 (relaxed for trade generation)
        if long_score >= 4:
            desired_signal = SIZE_LONG
        
        # SHORT ENTRIES
        if desired_signal == 0.0:
            short_score = 0
            
            # HTF bearish bias
            if htf_bearish:
                short_score += 2
            
            # Price below 4h KAMA
            if price_below_kama:
                short_score += 1
            
            # KAMA slope down
            if kama_slope_down:
                short_score += 1
            
            # RSI pullback (not oversold)
            if rsi_pullback_short:
                short_score += 1
            
            # ADX shows trend
            if trend_strong or htf_bearish:
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
                # Hold long if HTF still bullish OR price above KAMA
                if htf_bullish or price_above_kama:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HTF still bearish OR price below KAMA
                if htf_bearish or price_below_kama:
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