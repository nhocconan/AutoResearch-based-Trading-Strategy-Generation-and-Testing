#!/usr/bin/env python3
"""
Experiment #224: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + ADX Filter + RSI Pullback

Hypothesis: After 188 failed experiments, complex regime switching (chop/trend) consistently fails.
KAMA (Kaufman Adaptive Moving Average) adapts to volatility automatically - slows in chop, speeds in trend.
This is DIFFERENT from failed attempts: NO Choppiness Index, NO CRSI, NO dual regime switching.

Key differences from failed strategies:
1. KAMA instead of HMA/EMA - adapts to market conditions automatically
2. Simple ADX filter (only trade when ADX > 18) - avoids whipsaw
3. RSI pullback entries (not breakouts) - better risk/reward in bear markets
4. 12h HMA for macro bias (aligned via mtf_data)
5. 1d HMA for ultra-long-term trend filter
6. Discrete position sizing (0.0, ±0.20, ±0.30) to minimize fee churn

TARGET: 25-45 trades/year on 4h, Sharpe > 0.5 on ALL symbols
Position sizing: 0.0, ±0.20, ±0.30 (discrete levels)
Stoploss: ATR(14) 2.5x trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_rsi_12h1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market volatility - moves fast in trends, slow in chop.
    Efficiency Ratio (ER) determines smoothing constant.
    
    Reference: Kaufman, "Trading Systems and Methods"
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    # ER = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    change = np.abs(close_s.diff(period))
    volatility = close_s.diff().abs().rolling(window=period, min_periods=period).sum()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = change / (volatility + 1e-10)
    er = er.fillna(0).values
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if i < period:
            kama[i] = close_s.iloc[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) window
    Faster and smoother than EMA, less lag.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_atr = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_atr = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * plus_atr[i] / atr[i]
            minus_di[i] = 100.0 * minus_atr[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_series.values
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    kama_fast = calculate_kama(close, period=10, fast_period=2, slow_period=20)
    kama_slow = calculate_kama(close, period=40, fast_period=2, slow_period=30)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    # Calculate 12h HMA for intermediate trend (aligned properly)
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1d HMA for macro trend (aligned properly)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(adx_14[i]):
            continue
        
        # === HTF MACRO BIAS ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === TREND DETECTION (4h KAMA crossover) ===
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # KAMA slope confirmation
        kama_fast_slope_up = kama_fast[i] > kama_fast[i-1] if i > 0 else False
        kama_fast_slope_down = kama_fast[i] < kama_fast[i-1] if i > 0 else False
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx_14[i] > 18.0
        weak_trend = adx_14[i] <= 18.0
        
        # === RSI PULLBACK FILTER ===
        # For longs: RSI pulled back but not oversold (40-55)
        rsi_pullback_long = 35.0 <= rsi_14[i] <= 55.0
        rsi_not_overbought = rsi_14[i] < 65.0
        
        # For shorts: RSI rallied but not oversold (45-65)
        rsi_pullback_short = 45.0 <= rsi_14[i] <= 65.0
        rsi_not_oversold = rsi_14[i] > 35.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: KAMA bullish + ADX strong + RSI pullback + 12h bias
        if kama_bullish and kama_fast_slope_up:
            if strong_trend and rsi_pullback_long:
                if price_above_hma_12h:
                    # With intermediate trend
                    if price_above_hma_1d:
                        new_signal = POSITION_SIZE_FULL  # All HTF aligned
                    else:
                        new_signal = POSITION_SIZE_HALF  # Against 1d, smaller
                elif rsi_14[i] < 45.0:
                    # Deep pullback entry even against 12h
                    new_signal = POSITION_SIZE_HALF
        
        # SHORT ENTRY: KAMA bearish + ADX strong + RSI pullback + 12h bias
        elif kama_bearish and kama_fast_slope_down:
            if strong_trend and rsi_pullback_short:
                if price_below_hma_12h:
                    # With intermediate trend
                    if price_below_hma_1d:
                        new_signal = -POSITION_SIZE_FULL  # All HTF aligned
                    else:
                        new_signal = -POSITION_SIZE_HALF  # Against 1d, smaller
                elif rsi_14[i] > 55.0:
                    # Strong rally entry even against 12h
                    new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if KAMA still bullish and RSI not overbought
                if kama_bullish and rsi_14[i] < 70.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if KAMA still bearish and RSI not oversold
                if kama_bearish and rsi_14[i] > 30.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if KAMA crosses bearish
        if in_position and position_side > 0 and kama_bearish:
            new_signal = 0.0
        
        # Exit short if KAMA crosses bullish
        if in_position and position_side < 0 and kama_bullish:
            new_signal = 0.0
        
        # Exit if 12h trend reverses against position (medium-term filter)
        if in_position and position_side > 0 and price_below_hma_12h:
            new_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_12h:
            new_signal = 0.0
        
        # Exit if ADX drops too low (trend died)
        if in_position and adx_14[i] < 15.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals