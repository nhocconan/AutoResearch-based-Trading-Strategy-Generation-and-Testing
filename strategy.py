#!/usr/bin/env python3
"""
Experiment #561: 15m Primary + 1h/4h/1d HTF — Multi-Timeframe Trend Pullback

Hypothesis: 15m timeframe is underexplored (0 experiments). Using proven 4h HMA trend
direction with 1h confirmation and 15m RSI pullback entries should generate trades
while maintaining edge. Key insight from failures: entry conditions were TOO STRICT
(0 trades on most 15m strategies).

Key differences from failed 15m attempts:
1. LOOSEN entry thresholds: RSI 35/65 (not 25/75)
2. Fewer confluence requirements: 2 filters (not 3-4)
3. No strict session filter (allows trades 24/7 for crypto)
4. Smaller position size: 0.20-0.25 (appropriate for 15m frequency)
5. Simple ATR stoploss (2.5x) - proven to work

Strategy logic:
1. 4h HMA(21) = primary trend bias (load ONCE before loop)
2. 1h HMA(21) = intermediate confirmation (load ONCE before loop)
3. 1d HMA(21) = macro filter (load ONCE before loop)
4. 15m RSI(14) = entry timing (pullback in trend direction)
5. 15m ATR(14) = stoploss calculation

Entry conditions (LOOSENED for trade generation):
- LONG: 4h HMA bullish + price>4h HMA + RSI(14) crosses above 35 from below
- SHORT: 4h HMA bearish + price<4h HMA + RSI(14) crosses below 65 from above
- Add 1h HMA confirmation for stronger signals

Target: 40-100 trades/year, Sharpe>0.40, DD<-30%
Timeframe: 15m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_pullback_1h4h1d_loose_v1"
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
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1h HMA
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate and align 4h HMA
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    hma_15m = calculate_hma(close, period=21)
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h primary) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # 1h confirmation
        h1_bull = close[i] > hma_1h_aligned[i] if not np.isnan(hma_1h_aligned[i]) else False
        h1_bear = close[i] < hma_1h_aligned[i] if not np.isnan(hma_1h_aligned[i]) else False
        
        # 1d macro filter
        htf_macro_bull = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        htf_macro_bear = close[i] < hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        
        # === RSI PULLBACK LOGIC (LOOSENED) ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        rsi_recovering = rsi[i] > 35.0 and rsi[i-1] < 35.0 if i > 0 else False
        rsi_weakening = rsi[i] < 65.0 and rsi[i-1] > 65.0 if i > 0 else False
        
        # RSI in neutral zone (good for trend continuation)
        rsi_neutral_long = rsi[i] > 40.0 and rsi[i] < 60.0
        rsi_neutral_short = rsi[i] > 40.0 and rsi[i] < 60.0
        
        # === ENTRY LOGIC (LOOSENED for trade generation) ===
        desired_signal = 0.0
        
        # LONG entries - 4h bullish + RSI pullback
        if htf_bull:
            # Strong: 4h bull + 1h bull + RSI recovering from oversold
            if h1_bull and rsi_recovering:
                desired_signal = SIZE_STRONG
            # Base: 4h bull + RSI in neutral zone (trend continuation)
            elif rsi_neutral_long and htf_macro_bull:
                desired_signal = SIZE_BASE
            # RSI bounce from oversold in uptrend
            elif rsi_oversold and rsi[i] > rsi[i-1] if i > 0 else False:
                desired_signal = SIZE_BASE
        
        # SHORT entries - 4h bearish + RSI pullback
        elif htf_bear:
            # Strong: 4h bear + 1h bear + RSI weakening from overbought
            if h1_bear and rsi_weakening:
                desired_signal = -SIZE_STRONG
            # Base: 4h bear + RSI in neutral zone (trend continuation)
            elif rsi_neutral_short and htf_macro_bear:
                desired_signal = -SIZE_BASE
            # RSI drop from overbought in downtrend
            elif rsi_overbought and rsi[i] < rsi[i-1] if i > 0 else False:
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