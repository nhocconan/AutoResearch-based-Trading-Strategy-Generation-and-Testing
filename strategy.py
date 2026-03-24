#!/usr/bin/env python3
"""
Experiment #661: 15m Primary + 1h/4h HTF — HMA Trend + RSI Mean Reversion + HTF Bias

Hypothesis: 15m timeframe needs LOOSE entry conditions to generate trades (previous 15m 
strategies got Sharpe=0.000 = ZERO trades). Use simple confluence: HTF bias (1h/4h HMA) 
for direction, 15m RSI(7) extremes for entry timing, 15m HMA(21) for local trend filter.

Key innovations for 15m trade generation:
1. RSI(7) extremes (<30/>70) happen frequently = many entry opportunities
2. HTF (1h/4h) only filters DIRECTION, doesn't block entries
3. Only 2-3 confluence required (not 5+ like failed experiments)
4. No session filter (kills trade frequency)
5. Size 0.20-0.25 (smaller for 15m frequency, target 50-100 trades/year)
6. ATR(14) trailing stop at 2.5x for risk management

Entry conditions (LOOSE to ensure trades):
- LONG: 4h HMA bullish bias + RSI(7) < 35 + price > 15m HMA(21)
- SHORT: 4h HMA bearish bias + RSI(7) > 65 + price < 15m HMA(21)
- Exit: RSI crosses 50 OR stoploss hit

Target: Sharpe>0.40, trades>=50 train, trades>=5 test
Timeframe: 15m
Size: 0.20-0.25 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_1h4h_bias_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - responsive trend indicator"""
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
    delta = np.insert(delta, 0, 0.0)
    
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
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[avg_loss == 0] = 100.0
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF HMAs
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 15m indicators
    hma_15m = calculate_hma(close, period=21)
    rsi_15m = calculate_rsi(close, period=7)  # Fast RSI for frequent signals
    atr_15m = calculate_atr(high, low, close, period=14)
    
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
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_15m[i]) or atr_15m[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h HMA for major trend) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === 1H CONFIRMATION (intermediate trend) ===
        h1_bull = close[i] > hma_1h_aligned[i]
        h1_bear = close[i] < hma_1h_aligned[i]
        
        # === 15M LOCAL TREND ===
        local_bull = close[i] > hma_15m[i]
        local_bear = close[i] < hma_15m[i]
        
        # === RSI EXTREMES (entry trigger) ===
        rsi_oversold = rsi_15m[i] < 35.0
        rsi_overbought = rsi_15m[i] > 65.0
        
        # === ENTRY LOGIC (LOOSE - 2-3 confluence only) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + RSI oversold + (1h or 15m bullish)
        if htf_bull and rsi_oversold:
            if h1_bull or local_bull:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bearish + RSI overbought + (1h or 15m bearish)
        elif htf_bear and rsi_overbought:
            if h1_bear or local_bear:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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
        
        # === RSI EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and rsi_15m[i] > 55.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_15m[i] < 45.0:
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
                entry_atr = atr_15m[i]
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