#!/usr/bin/env python3
"""
Experiment #717: 1d Primary + 1w HTF — Donchian Breakout with HMA Trend Filter

Hypothesis: Daily breakouts work best when aligned with weekly trend direction.
Use 1w HMA(21) for major trend bias, 1d Donchian(20) for breakout entries,
RSI(14) to avoid overextended entries, ATR(14) for trailing stops.

Why this should work on 1d:
- 1d timeframe = fewer trades (20-50/year target), less fee drag
- Donchian breakout captures major moves (proven on SOL in history)
- 1w HMA filter prevents counter-trend breakouts (major failure mode)
- RSI filter avoids entering at extremes (reduces whipsaw)
- Simple logic = more trades (avoid 0-trade failures like #707, #712)

Key differences from failed 1d experiments:
- Simpler entry (Donchian breakout vs complex CRSI+Chop combos)
- Looser RSI filter (35-65 range, not narrow bands)
- 1w HTF for stronger trend bias (1d was too noisy in #713)
- Discrete signal sizes to minimize churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_1w_rsi_atr_v1"
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
    
    if n < period:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
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
    rsi_1d = calculate_rsi(close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_50 = calculate_sma(close, period=50)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
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
    
    for i in range(250, n):  # Need 200 for SMA + buffer for HTF alignment
        # Skip if indicators not ready
        if np.isnan(rsi_1d[i]) or np.isnan(atr_1d[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        if atr_1d[i] <= 1e-10:
            continue
        
        # === TREND BIAS (1w HTF HMA) ===
        trend_bullish = close[i] > hma_1w_aligned[i]
        trend_bearish = close[i] < hma_1w_aligned[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma50 = close[i] < sma_50[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI FILTER (avoid overextended) ===
        rsi_neutral = 35 < rsi_1d[i] < 65
        rsi_oversold = rsi_1d[i] < 40
        rsi_overbought = rsi_1d[i] > 60
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout = close crosses above upper or below lower
        breakout_long = close[i] > donchian_upper[i]
        breakout_short = close[i] < donchian_lower[i]
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRY CONDITIONS ===
        # Primary: Breakout + Bullish weekly trend + RSI not overbought
        if breakout_long and trend_bullish and rsi_1d[i] < 65:
            # Stronger signal if above SMA50
            if above_sma50:
                desired_signal = current_size
            else:
                desired_signal = REDUCED_SIZE
        
        # Secondary: Pullback long in bullish trend (RSI oversold + above weekly HMA)
        elif trend_bullish and rsi_oversold and above_sma200:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        # Primary: Breakdown + Bearish weekly trend + RSI not oversold
        if breakout_short and trend_bearish and rsi_1d[i] > 35:
            # Stronger signal if below SMA50
            if below_sma50:
                desired_signal = -current_size
            else:
                desired_signal = -REDUCED_SIZE
        
        # Secondary: Rally short in bearish trend (RSI overbought + below weekly HMA)
        elif trend_bearish and rsi_overbought and below_sma200:
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
                # Hold long if weekly trend still bullish and RSI not extreme
                if trend_bullish and rsi_1d[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if weekly trend still bearish and RSI not extreme
                if trend_bearish and rsi_1d[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if weekly trend reverses or RSI very overbought
            if trend_bearish or rsi_1d[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if weekly trend reverses or RSI very oversold
            if trend_bullish or rsi_1d[i] < 20:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE * 0.8 else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE * 0.8 else -REDUCED_SIZE
        else:
            desired_signal = 0.0
        
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