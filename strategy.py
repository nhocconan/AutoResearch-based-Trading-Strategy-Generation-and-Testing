#!/usr/bin/env python3
"""
Experiment #107: 6h Primary + 1w HTF — KAMA Trend + RSI Pullback + ADX Filter

Hypothesis: After 99 failed experiments, 6h strategies fail due to OVER-FILTERING.
Previous 6h attempts (#095, #100, #103) all had Sharpe < 0 due to:
- Too many conflicting regime filters (Choppiness + multiple HTF)
- Complex entry logic that rarely triggers
- Fisher Transform whipsaw on 6h timeframe

NEW APPROACH for 6h:
- SIMPLER logic: KAMA (adaptive) + RSI pullback + ADX trend strength
- HTF: 1w HMA (MORE STABLE than 1d, less whipsaw in bear markets)
- Entry: RSI pullback TO trend (not extremes) - generates MORE trades
- ADX > 20 filter (not 25+) - loose enough to trigger, strict enough to avoid chop
- Asymmetric sizing: 0.30 for trend-aligned, 0.20 for counter-trend
- This should generate 40-80 trades/year on 6h (within target 30-60)

Key design choices:
- Timeframe: 6h (untested, middle ground between 4h and 12h)
- HTF: 1w HMA(21) for major trend bias (weekly trend is more stable)
- Entry: KAMA(21) trend + RSI(14) pullback to 40-60 zone + ADX(14) > 20
- Position size: 0.30 max (30% of capital, conservative)
- Stoploss: 2.5x ATR(14) trailing
- LOOSE filters to ensure >=30 trades on train, >=3 on test

Target: Sharpe > 0.167 (beat current best), DD > -40%, trades >= 30 train, >= 3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_rsi_adx_1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=21, fast=2, slow=30):
    """
    Kaufman's Adaptive Moving Average (KAMA)
    Adapts to market volatility - smooth in chop, fast in trends
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = sum(abs(close[j] - close[j-1]) for j in range(i - period + 1, i + 1))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 25 = strong trend, ADX < 20 = range/chop
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range and DM
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0.0)
        else:
            plus_dm[i] = 0.0
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0.0)
        else:
            minus_dm[i] = 0.0
    
    # Smooth TR and DM
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period, n):
        if atr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / atr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / atr_smooth[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    for i in range(period * 2, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0
    
    # ADX is SMA of DX
    adx[period * 2:] = pd.Series(dx[period * 2:]).rolling(window=period, min_periods=period).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    kama = calculate_kama(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30  # 30% for trend-aligned trades
    SIZE_COUNTER = 0.20  # 20% for counter-trend (riskier)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === TREND STRENGTH (ADX) ===
        # ADX > 20 = some trend, ADX > 25 = strong trend
        adx_strong = adx[i] > 25.0
        adx_weak = adx[i] <= 20.0
        
        # === KAMA TREND DIRECTION ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === KAMA SLOPE (momentum) ===
        kama_slope_bull = kama[i] > kama[i-5] if i >= 5 else False
        kama_slope_bear = kama[i] < kama[i-5] if i >= 5 else False
        
        # === RSI PULLBACK ZONE (not extremes - generates MORE trades) ===
        # Long: RSI pulled back to 35-55 in uptrend
        # Short: RSI pulled back to 45-65 in downtrend
        rsi_pullback_long = 35.0 <= rsi[i] <= 55.0
        rsi_pullback_short = 45.0 <= rsi[i] <= 65.0
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        
        # === DESIRED SIGNAL (Simplified Logic) ===
        desired_signal = 0.0
        
        # LONG entries (multiple confluence, but LOOSE enough to trigger)
        if kama_bull and kama_slope_bull:
            if htf_bull and rsi_pullback_long and adx[i] > 20.0:
                # Best case: HTF bull + RSI pullback + some trend
                desired_signal = SIZE_TREND
            elif rsi_oversold and adx[i] > 15.0:
                # Fallback: RSI oversold + weak trend (counter-trend bounce)
                desired_signal = SIZE_COUNTER
            elif htf_bull and rsi[i] < 50.0:
                # Simple: HTF bull + RSI below 50 (pullback)
                desired_signal = SIZE_TREND * 0.8
        
        # SHORT entries
        elif kama_bear and kama_slope_bear:
            if htf_bear and rsi_pullback_short and adx[i] > 20.0:
                # Best case: HTF bear + RSI pullback + some trend
                desired_signal = -SIZE_TREND
            elif rsi_overbought and adx[i] > 15.0:
                # Fallback: RSI overbought + weak trend (counter-trend bounce)
                desired_signal = -SIZE_COUNTER
            elif htf_bear and rsi[i] > 50.0:
                # Simple: HTF bear + RSI above 50 (pullback)
                desired_signal = -SIZE_TREND * 0.8
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_TREND * 0.85:
            final_signal = SIZE_TREND
        elif desired_signal <= -SIZE_TREND * 0.85:
            final_signal = -SIZE_TREND
        elif desired_signal >= SIZE_COUNTER * 0.85:
            final_signal = SIZE_COUNTER
        elif desired_signal <= -SIZE_COUNTER * 0.85:
            final_signal = -SIZE_COUNTER
        elif abs(desired_signal) >= SIZE_TREND * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_TREND * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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
        
        signals[i] = final_signal
    
    return signals