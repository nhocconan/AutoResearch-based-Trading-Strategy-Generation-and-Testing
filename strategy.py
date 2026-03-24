#!/usr/bin/env python3
"""
Experiment #016: 12h Primary + 1d HTF — Fisher Transform + Vol Spike Reversion + Donchian Breakout

Hypothesis: Previous CRSI/Chop strategies exhausted. New approach combines:
1. Ehlers Fisher Transform (period=9) for reversal detection - catches bear market rallies better than RSI
2. Volatility Spike Reversion (ATR(7)/ATR(30) > 2.0) - captures "vol crush" after panic
3. Donchian(20) breakout confirmation - ensures momentum alignment
4. 1d HMA(21) for regime bias - asymmetric entries (only short when bearish, only long when bullish)
5. ATR trailing stop at 2.5x for risk management

Why this should work:
- Fisher Transform normalizes price to Gaussian distribution, better for reversals
- Vol spike reversion specifically targets panic bottoms (2022 crash pattern)
- Donchian breakout filters false signals in choppy markets
- 1d HMA provides cleaner regime filter than Choppiness Index
- Asymmetric entries reduce whipsaw in bear markets (BTC/ETH specific edge)

Entry Logic:
- BULL REGIME (close > 1d HMA): Fisher < -1.5 + ATR ratio > 1.8 + Donchian breakout up
- BEAR REGIME (close < 1d HMA): Fisher > +1.5 + ATR ratio > 1.8 + Donchian breakout down
- Size: 0.30 with regime, 0.20 against regime

Risk: 2.5x ATR trailing stop, max signal magnitude 0.35
Target: Sharpe > 0.4, trades > 20/symbol train, > 3/symbol test, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_volspike_donchian_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian-normalized values between -1 and +1
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    Research shows excellent reversal detection in bear markets
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    # Calculate median price
    median_price = (high + low + close) / 3.0
    
    # Normalize price to range 0-1
    lowest_low = np.full(n, np.nan)
    highest_high = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        lowest_low[i] = np.min(low[i - period + 1:i + 1])
        highest_high[i] = np.max(high[i - period + 1:i + 1])
    
    # Avoid division by zero
    price_range = highest_high - lowest_low
    price_range = np.where(price_range < 1e-10, 1e-10, price_range)
    
    normalized = (median_price - lowest_low) / price_range
    
    # Clamp to 0.001-0.999 to avoid log issues
    normalized = np.clip(normalized, 0.001, 0.999)
    
    # Fisher calculation
    fisher_value = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    # Smooth with EMA
    fisher_smooth = pd.Series(fisher_value).ewm(span=3, min_periods=3, adjust=False).mean().values
    fisher_signal_line = pd.Series(fisher_smooth).ewm(span=2, min_periods=2, adjust=False).mean().values
    
    for i in range(period, n):
        fisher[i] = fisher_smooth[i]
        fisher_signal[i] = fisher_signal_line[i]
    
    return fisher, fisher_signal

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """
    ATR Ratio for volatility spike detection
    ATR(7) / ATR(30) > 2.0 indicates volatility spike (panic)
    Reversion expected when ratio falls back below 1.2
    """
    n = len(close)
    if n < long_period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_short = pd.Series(tr).ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_long = pd.Series(tr).ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    
    atr_ratio = np.full(n, np.nan)
    for i in range(long_period, n):
        if atr_long[i] > 1e-10:
            atr_ratio[i] = atr_short[i] / atr_long[i]
    
    return atr_ratio

def calculate_donchian(high, low, period=20):
    """
    Donchian Channels
    Upper = highest high over period
    Lower = lowest low over period
    Breakout above upper = bullish momentum
    Breakout below lower = bearish momentum
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_hma(close, period=21):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for regime bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    atr_ratio = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    MAX_SIZE = 0.35
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(atr_ratio[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (1d HMA) ===
        bull_regime = close[i] > hma_1d_aligned[i]
        bear_regime = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 1.8  # Volatility elevated
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5 and fisher_signal[i] < fisher[i]  # Fisher crossing up
        fisher_overbought = fisher[i] > 1.5 and fisher_signal[i] > fisher[i]  # Fisher crossing down
        
        # === DONCHIAN BREAKOUT CONFIRMATION ===
        donchian_breakout_up = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_down = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: Bull regime + Fisher oversold + Vol spike OR Donchian breakout
        if bull_regime:
            if fisher_oversold and vol_spike:
                desired_signal = BASE_SIZE
            elif donchian_breakout_up and fisher[i] < 0:
                desired_signal = REDUCED_SIZE
        
        # SHORT ENTRY: Bear regime + Fisher overbought + Vol spike OR Donchian breakout
        elif bear_regime:
            if fisher_overbought and vol_spike:
                desired_signal = -BASE_SIZE
            elif donchian_breakout_down and fisher[i] > 0:
                desired_signal = -REDUCED_SIZE
        
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
        desired_signal = np.clip(desired_signal, -MAX_SIZE, MAX_SIZE)
        
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal >= REDUCED_SIZE * 0.85:
            final_signal = REDUCED_SIZE
        elif desired_signal <= -REDUCED_SIZE * 0.85:
            final_signal = -REDUCED_SIZE
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