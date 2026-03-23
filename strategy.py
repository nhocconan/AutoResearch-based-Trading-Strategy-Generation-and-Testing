#!/usr/bin/env python3
"""
Experiment #743: 1d Primary + 1w HTF — Simple HMA Trend + Donchian Breakout + ATR Stop

Hypothesis: After 497 failed strategies, clear patterns emerge:
1. Complex regime detection (Chop + CRSI) = negative Sharpe (#731-735)
2. Simple HMA + Donchian on 1d with 1w filter should work (proven template from #737 Sharpe=0.234)
3. 1d timeframe targets 20-50 trades/year — need loose entry filters to ensure frequency
4. Current best is 4h triple regime (Sharpe=0.612) — I'll use simpler 1d logic with 1w bias

Strategy design:
1. 1w HMA(21) for major trend bias (proven in best strategies)
2. 1d HMA(16/48) crossover for trend direction
3. 1d Donchian(20) breakout for entries (simple, generates trades)
4. 1d RSI(14) loose filter (30-70) for timing
5. 1d ATR(14) trailing stop 2.5x for risk management
6. Discrete signals: 0.0, ±0.25, ±0.30

Key differences from failed experiments:
- NO Choppiness Index (caused negative Sharpe in 6+ experiments)
- NO complex CRSI regime switching (failed repeatedly)
- Simple HMA crossover (proven in multiple positive Sharpe strategies)
- Loose RSI filters to ensure trade frequency (>=10 trades train)
- Clear hold logic to maintain positions through trends

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1d (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_donchian_rsi_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel for breakout detection."""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    hma_1d_fast = calculate_hma(close, 16)
    hma_1d_slow = calculate_hma(close, 48)
    rsi_1d = calculate_rsi(close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    sma_50 = calculate_sma(close, period=50)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):  # Need buffer for all indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(rsi_1d[i]) or np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        if np.isnan(hma_1d_fast[i]) or np.isnan(hma_1d_slow[i]):
            continue
        
        # === MAJOR TREND BIAS (1w HTF HMA) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === HMA CROSSOVER (1d) ===
        hma_bullish = hma_1d_fast[i] > hma_1d_slow[i]
        hma_bearish = hma_1d_fast[i] < hma_1d_slow[i]
        
        # === RSI FILTERS (loose to ensure trades) ===
        rsi_ok_long = rsi_1d[i] < 70 and rsi_1d[i] > 30
        rsi_ok_short = rsi_1d[i] < 70 and rsi_1d[i] > 30
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma50 = close[i] < sma_50[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (loose to ensure trades) ===
        long_signal = False
        
        # Path 1: Donchian breakout + 1w bullish + HMA bullish
        if close[i] > donch_upper[i-1] and trend_1w_bullish and hma_bullish:
            long_signal = True
        
        # Path 2: HMA bullish + Price > SMA50 + 1w bullish
        if hma_bullish and above_sma50 and trend_1w_bullish:
            long_signal = True
        
        # Path 3: Strong trend (above SMA50/200) + 1w bullish + RSI ok
        if above_sma50 and above_sma200 and trend_1w_bullish and rsi_ok_long:
            long_signal = True
        
        # Path 4: HMA bullish + RSI momentum (40-60) + 1w bullish
        if hma_bullish and 40 < rsi_1d[i] < 60 and trend_1w_bullish:
            long_signal = True
        
        if long_signal:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY CONDITIONS (loose to ensure trades) ===
        short_signal = False
        
        # Path 1: Donchian breakdown + 1w bearish + HMA bearish
        if close[i] < donch_lower[i-1] and trend_1w_bearish and hma_bearish:
            short_signal = True
        
        # Path 2: HMA bearish + Price < SMA50 + 1w bearish
        if hma_bearish and below_sma50 and trend_1w_bearish:
            short_signal = True
        
        # Path 3: Strong downtrend (below SMA50/200) + 1w bearish + RSI ok
        if below_sma50 and below_sma200 and trend_1w_bearish and rsi_ok_short:
            short_signal = True
        
        # Path 4: HMA bearish + RSI momentum (40-60) + 1w bearish
        if hma_bearish and 40 < rsi_1d[i] < 60 and trend_1w_bearish:
            short_signal = True
        
        if short_signal:
            desired_signal = -BASE_SIZE
        
        # === CONFLICT RESOLUTION ===
        # If both long and short signals, go with 1w HMA trend
        if long_signal and short_signal:
            if trend_1w_bullish:
                desired_signal = BASE_SIZE
            elif trend_1w_bearish:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = 0.0
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1w HMA still bullish and HMA still bullish
                if trend_1w_bullish and hma_bullish:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1w HMA still bearish and HMA still bearish
                if trend_1w_bearish and hma_bearish:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1w trend reverses or HMA reverses
            if trend_1w_bearish or hma_bearish:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1w trend reverses or HMA reverses
            if trend_1w_bullish or hma_bullish:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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
        
        signals[i] = desired_signal
    
    return signals