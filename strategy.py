#!/usr/bin/env python3
"""
Experiment #569: 15m Primary + 1h/1d HTF — Simple RSI Pullback + HMA Trend

Hypothesis: 15m strategies have FAILED (Sharpe=0.000) because entry conditions were TOO STRICT.
This strategy uses LOOSE entry conditions to ensure trades are generated:
1. 1h HMA(21) for trend direction (simple, proven)
2. 15m RSI(7) for entry timing (faster than RSI(14))
3. ATR volatility filter (avoid dead periods)
4. Session bias (00-12 UTC preferred but not required)
5. Discrete signal sizes: 0.0, ±0.15, ±0.20, ±0.25

Key differences from failed #561, #565:
1. LOOSER RSI thresholds (30/70 not 25/75)
2. Fewer HTF dependencies (1h only, not 1h+4h+1d)
3. No complex regime detection that blocks entries
4. Simple logic: HTF trend + RSI extreme = entry
5. ATR filter only skips VERY low vol (not moderate)

Target: Sharpe>0.40, trades>=120 train (30/year), trades>=15 test
Timeframe: 15m
Size: 0.15-0.25 (smaller for higher frequency on 15m)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_hma_1h_simple_loose_v1"
timeframe = "15m"
leverage = 1.0

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

def calculate_sma(close, period):
    """Simple Moving Average"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1h HMA for trend bias
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1h trend + 1d macro) ===
        htf_bull = close[i] > hma_1h_aligned[i]
        htf_bear = close[i] < hma_1h_aligned[i]
        
        # 1d macro confirmation (optional boost)
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY FILTER (avoid dead periods) ===
        atr_ratio = atr_14[i] / atr_30[i] if atr_30[i] > 1e-10 else 1.0
        vol_ok = atr_ratio > 0.5  # Only skip extremely low vol
        
        # === RSI EXTREMES (LOOSE thresholds for trades) ===
        rsi_oversold = rsi_7[i] < 35.0
        rsi_overbought = rsi_7[i] > 65.0
        rsi_extreme_oversold = rsi_7[i] < 25.0
        rsi_extreme_overbought = rsi_7[i] > 75.0
        
        # RSI recovery signals
        rsi_rising_from_oversold = rsi_oversold and i > 0 and rsi_7[i] > rsi_7[i-1]
        rsi_falling_from_overbought = rsi_overbought and i > 0 and rsi_7[i] < rsi_7[i-1]
        
        # === SESSION FILTER (UTC hour from open_time) ===
        # open_time is in milliseconds since epoch
        hour_utc = (open_time[i] // 3600000) % 24
        is_active_session = 0 <= hour_utc <= 12  # London/NY overlap preferred
        
        # === ENTRY LOGIC (LOOSE - ensure trades generated) ===
        desired_signal = 0.0
        
        # LONG entries
        if htf_bull and vol_ok:
            # Strong long: HTF bull + extreme oversold
            if rsi_extreme_oversold:
                desired_signal = SIZE_STRONG
            # Standard long: HTF bull + oversold + rising RSI
            elif rsi_oversold and rsi_rising_from_oversold:
                desired_signal = SIZE_BASE
            # Pullback long: HTF bull + RSI neutral but rising from low
            elif rsi_7[i] < 45.0 and rsi_rising_from_oversold:
                desired_signal = SIZE_BASE * 0.8
            # Macro confirmation boost
            if macro_bull and desired_signal > 0:
                desired_signal = min(desired_signal * 1.1, SIZE_STRONG)
        
        # SHORT entries
        elif htf_bear and vol_ok:
            # Strong short: HTF bear + extreme overbought
            if rsi_extreme_overbought:
                desired_signal = -SIZE_STRONG
            # Standard short: HTF bear + overbought + falling RSI
            elif rsi_overbought and rsi_falling_from_overbought:
                desired_signal = -SIZE_BASE
            # Pullback short: HTF bear + RSI neutral but falling from high
            elif rsi_7[i] > 55.0 and rsi_falling_from_overbought:
                desired_signal = -SIZE_BASE * 0.8
            # Macro confirmation boost
            if macro_bear and desired_signal < 0:
                desired_signal = max(desired_signal * 1.1, -SIZE_STRONG)
        
        # Session boost (prefer active hours but don't require)
        if is_active_session and abs(desired_signal) > 0:
            desired_signal = desired_signal * 1.1
            if desired_signal > 0:
                desired_signal = min(desired_signal, SIZE_STRONG)
            else:
                desired_signal = max(desired_signal, -SIZE_STRONG)
        
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