#!/usr/bin/env python3
"""
Experiment #152: 30m KAMA Adaptive Trend + 4h HMA Filter + RSI Pullback

Hypothesis: 30m timeframe balances trade frequency with signal quality. KAMA 
(Kaufman Adaptive Moving Average) adapts to market efficiency - fast in trends,
slow in ranges. Combined with 4h HMA trend filter and RSI pullback entries,
this should capture trend continuations while avoiding choppy whipsaws.

Why this might work where others failed:
- KAMA adapts to volatility unlike fixed EMA/HMA (critical for 2022 crash + 2025 bear)
- RSI pullback (not extreme) ensures entries WITH trend, not counter-trend
- Simple conditions avoid 0-trade problem (#142, #143 failures)
- 30m TF: faster than 4h/12h that failed, slower than 15m that struggled

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_4h_hma_rsi_pullback_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market efficiency ratio - fast in trends, slow in ranges.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio: measures trend vs noise
    # ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    change = np.abs(close - np.roll(close, period))
    change[0] = 0
    volatility = np.zeros(n)
    
    for i in range(period, n):
        vol_sum = 0.0
        for j in range(1, period + 1):
            if i - j >= 0:
                vol_sum += np.abs(close[i - j + 1] - close[i - j])
        volatility[i] = vol_sum if vol_sum > 0 else 1e-10
    
    er = np.zeros(n)
    er[period:] = change[period:] / volatility[period:]
    er[:period] = np.nan
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[period] = close[period]
    for i in range(period + 1, n):
        if np.isnan(sc[i]):
            continue
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(len(close))
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100.0  # No losses = RSI 100
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[avg_loss == 0] = 100.0
    
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, 14, 2, 30)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === KAMA TREND ON 30m ===
        # Price above KAMA = bullish, below = bearish
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === RSI PULLBACK LOGIC ===
        # In uptrend: enter on RSI pullback to 40-50 (not oversold extremes)
        # In downtrend: enter on RSI rally to 50-60 (not overbought extremes)
        rsi_pullback_long = 35 < rsi[i] < 55
        rsi_pullback_short = 45 < rsi[i] < 65
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # 4h bullish + 30m price > KAMA + RSI pullback (not extreme)
        if bull_trend_4h and kama_bull and rsi_pullback_long:
            new_signal = SIZE_BASE
        
        # Strong long: add momentum confirmation (RSI rising)
        if i > 1 and bull_trend_4h and kama_bull and rsi[i] > rsi[i-1] and rsi[i] > 50:
            new_signal = SIZE_STRONG
        
        # === SHORT ENTRY CONDITIONS ===
        # 4h bearish + 30m price < KAMA + RSI pullback (not extreme)
        if bear_trend_4h and kama_bear and rsi_pullback_short:
            new_signal = -SIZE_BASE
        
        # Strong short: add momentum confirmation (RSI falling)
        if i > 1 and bear_trend_4h and kama_bear and rsi[i] < rsi[i-1] and rsi[i] < 50:
            new_signal = -SIZE_STRONG
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals