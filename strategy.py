#!/usr/bin/env python3
"""
Experiment #719: 1h Primary + 4h/12h HTF — Regime-Adaptive RSI + Session Filter

Hypothesis: 1h timeframe with 4h trend bias + 12h regime filter provides optimal
trade frequency (40-80/year) with reduced whipsaw. Using Choppiness to switch
between trend-following and mean-reversion modes, with session filter for liquidity.

Key innovations:
1. 4h HMA(21) for trend direction (HTF bias)
2. 12h Choppiness(14) for regime detection (trend vs range)
3. 1h RSI(14) + Stochastic for entry timing
4. Session filter: 08-20 UTC (high liquidity hours)
5. ATR(14) 2.5x trailing stop
6. Discrete sizing: 0.0, ±0.20, ±0.30

Entry conditions (LOOSE to ensure trades):
- LONG trend: 4h HMA bull + CHOP<50 + RSI(14)>50 + session
- LONG mean revert: 4h HMA bull + CHOP>55 + RSI(14)<35 + session
- SHORT trend: 4h HMA bear + CHOP<50 + RSI(14)<50 + session
- SHORT mean revert: 4h HMA bear + CHOP>55 + RSI(14)>65 + session

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 1h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_rsi_session_4h12h_v1"
timeframe = "1h"
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - identifies trending vs ranging markets"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

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
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator"""
    n = len(close)
    if n < k_period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    k = np.zeros(n)
    k[:] = np.nan
    
    for i in range(k_period-1, n):
        lowest_low = low[i-k_period+1:i+1].min()
        highest_high = high[i-k_period+1:i+1].max()
        
        if highest_high == lowest_low:
            k[i] = 50.0
        else:
            k[i] = 100.0 * (close[i] - lowest_low) / (highest_high - lowest_low)
    
    d = pd.Series(k).ewm(span=d_period, min_periods=d_period, adjust=False).mean().values
    
    return k, d

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

def calculate_session_filter(open_time):
    """Session filter: 08-20 UTC (high liquidity hours)"""
    # open_time is in milliseconds
    hour = (open_time // 3600000) % 24
    return (hour >= 8) & (hour < 20)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    chop_12h_raw = calculate_choppiness(
        df_12h['high'].values,
        df_12h['low'].values,
        df_12h['close'].values,
        period=14
    )
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h_raw)
    
    # Calculate 1h indicators
    rsi_14 = calculate_rsi(close, period=14)
    stoch_k, stoch_d = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    atr = calculate_atr(high, low, close, period=14)
    session_active = calculate_session_filter(open_time)
    
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(stoch_k[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(chop_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (12h Choppiness) ===
        trend_regime = chop_12h_aligned[i] < 50.0  # Lower = trending
        range_regime = chop_12h_aligned[i] > 55.0  # Higher = ranging
        
        # === SESSION FILTER ===
        in_session = session_active[i]
        
        # === RSI CONDITIONS (LOOSE for trade generation) ===
        rsi_bull = rsi_14[i] > 50.0
        rsi_bear = rsi_14[i] < 50.0
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === STOCHASTIC CONFIRMATION ===
        stoch_bull = stoch_k[i] > 50.0
        stoch_bear = stoch_k[i] < 50.0
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: Trend regime + HTF bull + RSI bull + session
        if trend_regime and htf_4h_bull and rsi_bull and in_session:
            desired_signal = SIZE_STRONG
        # LONG: Range regime + HTF bull + RSI oversold + session
        elif range_regime and htf_4h_bull and rsi_oversold and in_session:
            desired_signal = SIZE_BASE
        # LONG: HTF bull + RSI very oversold (any regime) + session
        elif htf_4h_bull and rsi_14[i] < 25.0 and in_session:
            desired_signal = SIZE_BASE
        # LONG: HTF bull + Stochastic bull + session (backup entry)
        elif htf_4h_bull and stoch_bull and in_session and rsi_14[i] > 45.0:
            desired_signal = SIZE_BASE
        
        # SHORT: Trend regime + HTF bear + RSI bear + session
        elif trend_regime and htf_4h_bear and rsi_bear and in_session:
            desired_signal = -SIZE_STRONG
        # SHORT: Range regime + HTF bear + RSI overbought + session
        elif range_regime and htf_4h_bear and rsi_overbought and in_session:
            desired_signal = -SIZE_BASE
        # SHORT: HTF bear + RSI very overbought (any regime) + session
        elif htf_4h_bear and rsi_14[i] > 75.0 and in_session:
            desired_signal = -SIZE_BASE
        # SHORT: HTF bear + Stochastic bear + session (backup entry)
        elif htf_4h_bear and stoch_bear and in_session and rsi_14[i] < 55.0:
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