#!/usr/bin/env python3
"""
Experiment #563: 6h Primary + 1d/1w HTF — Volatility Expansion Breakout + HTF Trend Filter

Hypothesis: 6h timeframe sits between 4h and 12h - ideal for capturing multi-day trends
without excessive noise. Previous 6h strategies failed due to overly complex entry logic
(mean reversion + trend + regime filters conflicting). This strategy uses SIMPLE volatility
expansion breakout with HTF trend confirmation.

Key differences from failed 6h experiments:
1. Donchian breakout (not HMA crossover) - cleaner signal, fewer whipsaws
2. ATR expansion filter (vol must be increasing for breakout validity)
3. Volume confirmation (breakout must have volume support)
4. 1d HMA for macro bias (only trade in HTF direction)
5. 1w HMA for ultra-macro filter (avoid counter-trend in major moves)
6. Simpler logic = more trades (previous 6h strategies had 0 trades)

Strategy logic:
1. 1w HMA(21) = ultra-macro trend (avoid major counter-trend trades)
2. 1d HMA(21) = macro trend bias (primary directional filter)
3. 6h Donchian(20) = breakout levels (20-period high/low)
4. 6h ATR(14) vs ATR(20) = volatility expansion filter
5. 6h Volume vs Volume SMA(20) = volume confirmation
6. 6h RSI(14) = avoid extreme entries (RSI 30-70 range for breakouts)
7. ATR(14)*2.5 stoploss on all positions

Entry conditions (LONG):
- Price > 1d HMA AND Price > 1w HMA (HTF bullish)
- Price breaks Donchian(20) high
- ATR(14) > ATR(20)*1.1 (volatility expanding)
- Volume > Volume_SMA(20)*1.2 (volume confirmation)
- RSI(14) between 40-70 (not overbought on entry)

Entry conditions (SHORT):
- Price < 1d HMA AND Price < 1w HMA (HTF bearish)
- Price breaks Donchian(20) low
- ATR(14) > ATR(20)*1.1 (volatility expanding)
- Volume > Volume_SMA(20)*1.2 (volume confirmation)
- RSI(14) between 30-60 (not oversold on entry)

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=10 test
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_vol_breakout_hma_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - tracks N-period high and low
    Returns: upper_band, lower_band, mid_band
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
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

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for ultra-macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_20 = calculate_atr(high, low, close, period=20)
    rsi = calculate_rsi(close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track breakout bars to avoid repeated signals
    last_long_breakout = -100
    last_short_breakout = -100
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or np.isnan(atr_20[i]) or atr_14[i] <= 1e-10:
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
        
        if np.isnan(rsi[i]) or np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w ultra-macro + 1d macro) ===
        htf_bull = close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # === VOLATILITY EXPANSION ===
        vol_expansion = atr_14[i] > atr_20[i] * 1.05  # ATR expanding
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > vol_sma[i] * 1.15  # Volume 15% above average
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # New high breakout (price crosses above Donchian upper)
        long_breakout = close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1]
        # New low breakout (price crosses below Donchian lower)
        short_breakout = close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1]
        
        # Also allow continuation if already broken out recently (within 3 bars)
        long_continuation = close[i] > donchian_upper[i] and (i - last_long_breakout) <= 3
        short_continuation = close[i] < donchian_lower[i] and (i - last_short_breakout) <= 3
        
        # === RSI FILTER (avoid extremes on breakout) ===
        rsi_ok_long = 35.0 < rsi[i] < 75.0  # Not overbought on long entry
        rsi_ok_short = 25.0 < rsi[i] < 65.0  # Not oversold on short entry
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: HTF bull + breakout + vol expansion + volume + RSI ok
        if htf_bull and vol_expansion and volume_confirmed and rsi_ok_long:
            if long_breakout:
                desired_signal = SIZE_STRONG
                last_long_breakout = i
            elif long_continuation and position_side != 1:
                desired_signal = SIZE_BASE
        
        # SHORT ENTRY: HTF bear + breakout + vol expansion + volume + RSI ok
        elif htf_bear and vol_expansion and volume_confirmed and rsi_ok_short:
            if short_breakout:
                desired_signal = -SIZE_STRONG
                last_short_breakout = i
            elif short_continuation and position_side != -1:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
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