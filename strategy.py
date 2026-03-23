#!/usr/bin/env python3
"""
Experiment #217: 1d Primary + 1w HTF — KAMA Adaptive Trend + Donchian Breakout + ADX Filter

Hypothesis: After 12h failures with complex regime-switching (#206, #212, #213, #214),
shift to 1d primary timeframe with simpler KAMA adaptive trend system. KAMA automatically
adjusts to market volatility (fast in trends, slow in chop), reducing whipsaws without
needing explicit regime detection. Research showed KAMA+ADX+Choppiness worked on ETH
(Sharpe +0.755), and Donchian breakout + HMA worked on SOL (Sharpe +0.782).

Key design:
1. KAMA(14) adaptive trend — no need for separate chop filter, KAMA handles it
2. ADX(14) > 25 for trend strength confirmation (avoid choppy entries)
3. Donchian(20) breakout for entry timing
4. 1w KAMA(21) for macro bias (aligned via mtf_data)
5. RSI(14) filter to avoid extreme entries
6. ATR(14) 2.5x trailing stoploss

TARGET: 20-40 trades/year on 1d, Sharpe > 0.5 on ALL symbols
Position sizing: 0.0, ±0.30 (discrete to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_donchian_adx_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise: fast in trends, slow in chop.
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(1, n):
        if i < period:
            kama[i] = close[i]
            continue
        
        # Efficiency Ratio
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        
        if noise > 1e-10:
            er = signal / noise
        else:
            er = 0.0
        
        # Smoothing Constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    kama_14 = calculate_kama(close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    # Calculate 1w KAMA for macro trend (aligned properly)
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=21)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30
    
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
        if np.isnan(kama_14[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        if np.isnan(kama_1w_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(adx_14[i]):
            continue
        
        # === HTF MACRO BIAS (1w KAMA) ===
        price_above_kama_1w = close[i] > kama_1w_aligned[i]
        price_below_kama_1w = close[i] < kama_1w_aligned[i]
        
        # === TREND DETECTION (1d KAMA slope) ===
        kama_bullish = kama_14[i] > kama_14[i - 5] if i >= 5 else False
        kama_bearish = kama_14[i] < kama_14[i - 5] if i >= 5 else False
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama_14[i]
        price_below_kama = close[i] < kama_14[i]
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx_14[i] > 25.0
        weak_trend = adx_14[i] <= 25.0
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i - 1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i - 1] if i > 0 else False
        
        # === RSI FILTER (avoid extremes) ===
        rsi_bullish_ok = rsi_14[i] < 70.0
        rsi_bearish_ok = rsi_14[i] > 30.0
        rsi_neutral = 35.0 <= rsi_14[i] <= 65.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: KAMA bullish + Donchian breakout + ADX strong + RSI ok + 1w bias
        if kama_bullish and price_above_kama:
            if breakout_long and rsi_bullish_ok:
                if strong_trend:
                    if price_above_kama_1w:
                        new_signal = POSITION_SIZE  # All conditions met
                    else:
                        new_signal = POSITION_SIZE * 0.5  # Against 1w bias
                elif rsi_neutral and price_above_kama_1w:
                    new_signal = POSITION_SIZE * 0.5  # Pullback entry with macro
        
        # SHORT ENTRY: KAMA bearish + Donchian breakout + ADX strong + RSI ok + 1w bias
        elif kama_bearish and price_below_kama:
            if breakout_short and rsi_bearish_ok:
                if strong_trend:
                    if price_below_kama_1w:
                        new_signal = -POSITION_SIZE  # All conditions met
                    else:
                        new_signal = -POSITION_SIZE * 0.5  # Against 1w bias
                elif rsi_neutral and price_below_kama_1w:
                    new_signal = -POSITION_SIZE * 0.5  # Pullback entry with macro
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if KAMA still bullish and RSI not overbought
                if kama_bullish and rsi_14[i] < 75.0:
                    new_signal = signals[i - 1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if KAMA still bearish and RSI not oversold
                if kama_bearish and rsi_14[i] > 25.0:
                    new_signal = signals[i - 1] if i > 0 else 0.0
        
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
        
        # Exit if macro trend reverses against position
        if in_position and position_side > 0 and price_below_kama_1w and weak_trend:
            new_signal = 0.0
        
        if in_position and position_side < 0 and price_above_kama_1w and weak_trend:
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