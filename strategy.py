#!/usr/bin/env python3
"""
Experiment #1012: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend + Donchian Breakout + ADX Filter

Hypothesis: Kaufman Adaptive Moving Average (KAMA) automatically adjusts sensitivity based on 
market efficiency ratio, making it superior to fixed EMAs in both trending and ranging markets.
Combined with Donchian channel breakouts for entry timing and ADX filter to avoid choppy markets,
this should capture major trends while minimizing whipsaws.

Key innovations:
1. KAMA (Efficiency Ratio adaptive): ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
   - Fast SC = 2/(2+1) when trending, Slow SC = 2/(30+1) when ranging
   - Automatically smooths in chop, responds in trends
2. Donchian Channel(20): Breakout above 20-bar high or below 20-bar low
   - Proven trend-following entry (Turtle Trading system)
3. ADX(14) filter: Only trade when ADX > 15 (market has directional movement)
4. HTF alignment: 1d KAMA(21) for intermediate bias, 1w KAMA(21) for long-term trend
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.20, ±0.30 to minimize fee churn

Why 12h should work:
- Captures multi-day to weekly trends without 4h noise
- 20-50 trades/year target (lower fee drag than lower TFs)
- KAMA adapts to 2022 crash (ranging) and 2021/2024 bull (trending)
- Donchian breakouts catch major moves early

Entry conditions (LOOSE to guarantee trades):
- LONG: ADX>15 + price>Donchian_high(20) + price>1d_KAMA>1w_KAMA + RSI(14)>45
- SHORT: ADX>15 + price<Donchian_low(20) + price<1d_KAMA<1w_KAMA + RSI(14)<55

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 12h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_donchian_adx_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market Efficiency Ratio (ER)
    
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    
    Fast SC = 2/(fast_period+1), Slow SC = 2/(slow_period+1)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

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

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 25 = strong trend, ADX < 20 = ranging
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
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
    
    # Smooth TR, +DM, -DM
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI and DX
    plus_di = np.divide(plus_di, atr, out=np.zeros_like(plus_di), where=atr != 0) * 100.0
    minus_di = np.divide(minus_di, atr, out=np.zeros_like(minus_di), where=atr != 0) * 100.0
    
    di_sum = plus_di + minus_di
    dx = np.divide(np.abs(plus_di - minus_di), di_sum, out=np.zeros_like(plus_di), where=di_sum != 0) * 100.0
    
    # ADX = smoothed DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - tracks highest high and lowest low over period
    Returns: (upper_channel, lower_channel, middle_channel)
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    middle = (upper + lower) / 2.0
    return upper, lower, middle

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF KAMA indicators
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=21)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, period=20)
    kama_14 = calculate_kama(close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
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
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_14[i]) or np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND STRENGTH FILTER (ADX) ===
        is_trending = adx_14[i] > 15.0  # Market has directional movement
        
        # === HTF BIAS (KAMA alignment) ===
        kama_1d_bull = close[i] > kama_1d_aligned[i]
        kama_1d_bear = close[i] < kama_1d_aligned[i]
        kama_1w_bull = close[i] > kama_1w_aligned[i]
        kama_1w_bear = close[i] < kama_1w_aligned[i]
        
        # Strong trend alignment (all TFs agree)
        strong_bull = kama_1d_bull and kama_1w_bull and kama_1d_aligned[i] > kama_1w_aligned[i]
        strong_bear = kama_1d_bear and kama_1w_bear and kama_1d_aligned[i] < kama_1w_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        breakout_long = close[i] > donchian_upper[i - 1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i - 1] if i > 0 else False
        
        # === ENTRY LOGIC (TREND FOLLOWING) ===
        desired_signal = 0.0
        
        if is_trending:
            # LONG: Donchian breakout + HTF bullish + RSI confirmation
            if breakout_long and strong_bull and rsi_14[i] > 45.0 and rsi_14[i] < 80.0:
                desired_signal = SIZE_STRONG
            elif breakout_long and kama_1d_bull and kama_1w_bull and rsi_14[i] > 40.0:
                desired_signal = SIZE_BASE
            
            # SHORT: Donchian breakout + HTF bearish + RSI confirmation
            elif breakout_short and strong_bear and rsi_14[i] < 55.0 and rsi_14[i] > 20.0:
                desired_signal = -SIZE_STRONG
            elif breakout_short and kama_1d_bear and kama_1w_bear and rsi_14[i] < 60.0:
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