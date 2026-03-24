#!/usr/bin/env python3
"""
Experiment #652: 12h Primary + 1d HTF — HMA Trend + RSI Pullback + Donchian Breakout

Hypothesis: 12h timeframe with proven patterns from research:
1. 1d HMA(21) for HTF bias - long only above, short only below
2. 12h RSI(14) pullback entries - RSI<45 for long, RSI>55 for short (LOOSE)
3. Donchian(20) breakout confirmation - ensures momentum exists
4. ATR(14) volatility filter - avoid dead markets (ATR/price > 0.005)
5. Asymmetric sizing - 0.30 with HTF, 0.20 against HTF
6. ATR(14)*2.5 trailing stop - tight risk management

Key innovations vs failed experiments:
- LOOSER RSI thresholds (45/55 not 30/70) to ensure trades happen
- Donchian breakout adds momentum confirmation without blocking entries
- Volatility filter prevents trades in dead markets
- Discrete sizing: 0.0, ±0.20, ±0.30 to minimize fee churn

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-50%
Timeframe: 12h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_donchian_1d_v1"
timeframe = "12h"
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
    
    rs = np.zeros(n)
    rs[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100.0
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        rsi[i] = 100.0 - (100.0 / (1.0 + rs[i]))
    
    return rsi

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure"""
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
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    hma_12h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_WITH_HTF = 0.30
    SIZE_WITHOUT_HTF = 0.20
    
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(hma_12h[i]):
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLATILITY FILTER (avoid dead markets) ===
        vol_ratio = atr[i] / close[i]
        vol_ok = vol_ratio > 0.005  # ATR > 0.5% of price
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 12h HMA TREND ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === RSI PULLBACK (LOOSE thresholds for trades) ===
        rsi_oversold = rsi[i] < 45.0  # Long entry zone
        rsi_overbought = rsi[i] > 55.0  # Short entry zone
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] >= donchian_upper[i] * 0.995  # Near high
        breakout_short = close[i] <= donchian_lower[i] * 1.005  # Near low
        
        # === ENTRY LOGIC (LOOSE CONDITIONS - ensure trades) ===
        desired_signal = 0.0
        
        # LONG entries (multiple paths to ensure trades)
        if vol_ok:
            # Strong long: HTF bull + HMA bull + RSI pullback
            if htf_bull and hma_bull and rsi_oversold:
                desired_signal = SIZE_WITH_HTF
            # Medium long: HTF bull + RSI pullback + near breakout
            elif htf_bull and rsi_oversold and breakout_long:
                desired_signal = SIZE_WITH_HTF
            # Weaker long: HMA bull + RSI pullback (no HTF confirmation)
            elif hma_bull and rsi_oversold:
                desired_signal = SIZE_WITHOUT_HTF
            # Breakout long: Donchian breakout + HMA bull
            elif breakout_long and hma_bull:
                desired_signal = SIZE_WITHOUT_HTF
        
        # SHORT entries (multiple paths to ensure trades)
        if vol_ok:
            # Strong short: HTF bear + HMA bear + RSI overbought
            if htf_bear and hma_bear and rsi_overbought:
                desired_signal = -SIZE_WITH_HTF
            # Medium short: HTF bear + RSI overbought + near breakout
            elif htf_bear and rsi_overbought and breakout_short:
                desired_signal = -SIZE_WITH_HTF
            # Weaker short: HMA bear + RSI overbought (no HTF confirmation)
            elif hma_bear and rsi_overbought:
                desired_signal = -SIZE_WITHOUT_HTF
            # Breakout short: Donchian breakout + HMA bear
            elif breakout_short and hma_bear:
                desired_signal = -SIZE_WITHOUT_HTF
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop) if stop_price > 0 else trailing_stop
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop) if stop_price > 0 else trailing_stop
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_WITH_HTF * 0.9:
            final_signal = SIZE_WITH_HTF
        elif desired_signal <= -SIZE_WITH_HTF * 0.9:
            final_signal = -SIZE_WITH_HTF
        elif desired_signal >= SIZE_WITHOUT_HTF * 0.9:
            final_signal = SIZE_WITHOUT_HTF
        elif desired_signal <= -SIZE_WITHOUT_HTF * 0.9:
            final_signal = -SIZE_WITHOUT_HTF
        elif abs(desired_signal) >= SIZE_WITHOUT_HTF * 0.4:
            final_signal = np.sign(desired_signal) * SIZE_WITHOUT_HTF
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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