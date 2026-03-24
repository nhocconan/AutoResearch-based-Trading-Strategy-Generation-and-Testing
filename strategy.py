#!/usr/bin/env python3
"""
Experiment #815: 6h Primary + 12h/1d HTF — Donchian Breakout with Dual-HTF Confirmation

Hypothesis: 6h Donchian(20) breakouts filtered by dual-HTF trend alignment (12h + 1d)
will capture sustained moves while avoiding whipsaws. Previous 6h strategies failed
due to EMA crossover whipsaw or overly complex regime filters. Donchian breakouts
have cleaner entry/exit logic and work better in trending regimes.

Key innovations:
1. 6h Donchian(20) breakout for entries (not EMA crossover)
2. Dual-HTF confirmation: 12h HMA + 1d HMA must agree on direction
3. 6h RSI(14) as entry timing filter (avoid entering at extremes)
4. Donchian mid-line exit + ATR trailing stop
5. Asymmetric sizing: 0.30 when both HTF aligned, 0.20 when only 1d aligned
6. Discrete levels: 0.0, ±0.20, ±0.30

Entry conditions:
- LONG: 1d HMA bull + 12h HMA bull + price breaks Donchian(20) high + RSI < 70
- SHORT: 1d HMA bear + 12h HMA bear + price breaks Donchian(20) low + RSI > 30

Exit conditions:
- Price crosses Donchian mid-line against position
- ATR trailing stop hit (2.5x)
- RSI extreme (>80 long, <20 short)

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_hma_dual_12h1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout system"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    mid = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    mid[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        mid[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_WEAK = 0.20
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
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h + 1d HMA) ===
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Dual HTF alignment
        htf_both_bull = htf_12h_bull and htf_1d_bull
        htf_both_bear = htf_12h_bear and htf_1d_bear
        htf_mixed = not htf_both_bull and not htf_both_bear
        
        # === DONCHIAN BREAKOUT DETECTION ===
        donchian_breakout_long = False
        donchian_breakout_short = False
        
        if i > 0 and not np.isnan(donchian_upper[i-1]) and not np.isnan(donchian_lower[i-1]):
            # Breakout: price closes above upper (for long) or below lower (for short)
            donchian_breakout_long = (close[i-1] <= donchian_upper[i-1]) and (close[i] > donchian_upper[i])
            donchian_breakout_short = (close[i-1] >= donchian_lower[i-1]) and (close[i] < donchian_lower[i])
        
        # === RSI FILTERS (avoid entering at extremes) ===
        rsi_not_overbought = rsi_14[i] < 70.0
        rsi_not_oversold = rsi_14[i] > 30.0
        rsi_extreme_overbought = rsi_14[i] > 80.0
        rsi_extreme_oversold = rsi_14[i] < 20.0
        
        # === DONCHIAN MID-LINE EXIT SIGNAL ===
        exit_long = False
        exit_short = False
        
        if in_position and position_side > 0:
            if close[i] < donchian_mid[i]:
                exit_long = True
        
        if in_position and position_side < 0:
            if close[i] > donchian_mid[i]:
                exit_short = True
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Both HTF bull + Donchian breakout + RSI not overbought
        if htf_both_bull and donchian_breakout_long and rsi_not_overbought:
            desired_signal = SIZE_STRONG
        elif htf_1d_bull and donchian_breakout_long and rsi_not_overbought:
            # Only 1d aligned (weaker signal)
            desired_signal = SIZE_WEAK
        
        # SHORT: Both HTF bear + Donchian breakout + RSI not oversold
        elif htf_both_bear and donchian_breakout_short and rsi_not_oversold:
            desired_signal = -SIZE_STRONG
        elif htf_1d_bear and donchian_breakout_short and rsi_not_oversold:
            # Only 1d aligned (weaker signal)
            desired_signal = -SIZE_WEAK
        
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
        
        # === RSI EXTREME EXIT ===
        if in_position and position_side > 0 and rsi_extreme_overbought:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_extreme_oversold:
            desired_signal = 0.0
        
        # === DONCHIAN MID-LINE EXIT ===
        if exit_long or exit_short:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_WEAK * 0.9:
            final_signal = SIZE_WEAK
        elif desired_signal <= -SIZE_WEAK * 0.9:
            final_signal = -SIZE_WEAK
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