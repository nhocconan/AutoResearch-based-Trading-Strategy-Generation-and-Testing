#!/usr/bin/env python3
"""
Experiment #989: 4h Primary + 1d HTF — Donchian Breakout + HMA Trend + RSI Momentum

Hypothesis: After 700+ failed strategies, simplifying to ONE strong signal type (Donchian breakout)
with HTF trend filter (1d HMA) and momentum confirmation (RSI) will generate consistent trades
across ALL symbols while maintaining positive Sharpe.

Why this should work:
1. Donchian(20) breakout = clear, objective signal that triggers regularly
2. 1d HMA(21) = macro trend filter (only long in bull, short in bear)
3. RSI(14) 45-65 range = momentum confirmation without extreme conditions
4. 4h timeframe = 25-40 trades/year target (low fee drag)
5. Simple logic = less overfitting, works in both bull/bear markets

Key improvements over failed strategies:
- NO complex regime switching (choppiness, Fisher, etc. all failed)
- NO funding rate dependency (data alignment issues)
- RELAXED RSI conditions (45-65 not 30-70) to ensure trades trigger
- Clear stoploss at 2.5x ATR trailing
- Discrete signal sizes (0.0, ±0.25, ±0.30)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_breakout_1d_hma_rsi_momentum_atr_v1"
timeframe = "4h"
leverage = 1.0

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

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

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

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_4h_50 = calculate_sma(close, 50)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_4h_50[i]):
            continue
        
        # === MACRO TREND (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous high
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous low
        
        # === RSI MOMENTUM CONFIRMATION (relaxed for more trades) ===
        rsi_bullish = 45 < rsi_4h[i] < 70  # Momentum but not overbought
        rsi_bearish = 30 < rsi_4h[i] < 55  # Momentum but not oversold
        
        # === SMA50 FILTER (additional trend confirmation) ===
        price_above_sma50 = close[i] > sma_4h_50[i]
        price_below_sma50 = close[i] < sma_4h_50[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        if macro_bull:  # Only long in bull macro trend
            # Primary: Donchian breakout + RSI momentum + above SMA50
            if breakout_long and rsi_bullish and price_above_sma50:
                desired_signal = BASE_SIZE
            # Secondary: Donchian breakout + RSI momentum (less strict)
            elif breakout_long and rsi_bullish:
                desired_signal = REDUCED_SIZE
            # Tertiary: RSI pullback in bull trend (mean reversion entry)
            elif rsi_4h[i] < 40 and macro_bull and price_above_sma50:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        if macro_bear:  # Only short in bear macro trend
            # Primary: Donchian breakdown + RSI momentum + below SMA50
            if breakout_short and rsi_bearish and price_below_sma50:
                desired_signal = -BASE_SIZE
            # Secondary: Donchian breakdown + RSI momentum (less strict)
            elif breakout_short and rsi_bearish:
                desired_signal = -REDUCED_SIZE
            # Tertiary: RSI rally in bear trend (mean reversion entry)
            elif rsi_4h[i] > 60 and macro_bear and price_below_sma50:
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro bull and RSI not overbought
                if macro_bull and rsi_4h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro bear and RSI not oversold
                if macro_bear and rsi_4h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro trend reverses
            if macro_bear and rsi_4h[i] > 60:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro trend reverses
            if macro_bull and rsi_4h[i] < 40:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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