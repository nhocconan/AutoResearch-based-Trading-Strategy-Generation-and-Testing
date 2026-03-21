#!/usr/bin/env python3
"""
Experiment #061: 15m Fisher Transform with 4h HMA Trend Filter
Hypothesis: Fisher Transform excels at catching reversals in bear/range markets (2025 test period).
Combine with 4h HMA for trend direction filter to avoid counter-trend trades.
15m timeframe provides enough signals while 4h filter reduces whipsaws.
Fisher entry: long when crosses above -1.5 (oversold), short when crosses below +1.5 (overbought).
Position sizing: 0.25 entry, 0.15 at 2R profit, 2*ATR trailing stop.
This differs from failed RSI strategies by using Fisher's Gaussian normalization for better reversal detection.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_fisher_4h_hma_trend_v1"
timeframe = "15m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = period // 2
    if half < 1:
        half = 1
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into Gaussian normal distribution for better reversal detection.
    Entry when Fisher crosses -1.5 (long) or +1.5 (short).
    """
    hl2 = (high + low) / 2
    
    # Normalize price to -1 to +1 range
    highest = pd.Series(hl2).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(hl2).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = np.where(range_val < 0.0001, 0.0001, range_val)
    
    normalized = (hl2 - lowest) / range_val
    normalized = np.clip(normalized, 0.001, 0.999)  # Avoid log(0) or log(1)
    
    # Fisher transform
    fisher_input = 0.5 * np.log((1 + normalized) / (1 - normalized + 0.0001))
    fisher_input = np.nan_to_num(fisher_input, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Smooth with EMA
    fisher = pd.Series(fisher_input).ewm(span=period, min_periods=period, adjust=False).mean().values
    fisher = np.nan_to_num(fisher, nan=0.0)
    
    return fisher

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    fisher = calculate_fisher_transform(high, low, close, 9)
    rsi = calculate_rsi(close, 14)
    
    # 15m HMA for local trend confirmation
    hma_15m_21 = calculate_hma(close, 21)
    hma_15m_50 = calculate_hma(close, 50)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    
    for i in range(100, n):
        # 4h trend filter (HTF)
        hma_4h_valid = hma_4h_aligned[i] > 0 and not np.isnan(hma_4h_aligned[i])
        trend_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # Fisher Transform signals
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # Fisher cross signals (look at previous bar for cross detection)
        fisher_cross_long = (i > 1) and (fisher[i] > -1.5) and (fisher[i-1] <= -1.5)
        fisher_cross_short = (i > 1) and (fisher[i] < 1.5) and (fisher[i-1] >= 1.5)
        
        # 15m local trend
        local_bullish = hma_15m_21[i] > hma_15m_50[i]
        local_bearish = hma_15m_21[i] < hma_15m_50[i]
        
        # Volume confirmation (above average)
        vol_confirmed = volume[i] > vol_sma[i] if vol_sma[i] > 0 else True
        
        # RSI filter (avoid extreme overbought/oversold for entries)
        rsi_ok_long = rsi[i] < 70
        rsi_ok_short = rsi[i] > 30
        
        new_signal = 0.0
        
        # LONG ENTRY: Fisher cross + 4h bullish trend + volume confirmation
        if fisher_cross_long and trend_bullish and vol_confirmed and rsi_ok_long:
            new_signal = SIZE_ENTRY
        elif fisher_oversold and trend_bullish and local_bullish and vol_confirmed:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: Fisher cross + 4h bearish trend + volume confirmation
        if fisher_cross_short and trend_bearish and vol_confirmed and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        elif fisher_overbought and trend_bearish and local_bearish and vol_confirmed:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Calculate trailing stop
            current_stop = close[i] - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    profit = close[i] - entry_price
                    risk = 2.0 * atr[int(i)] if i > 0 else atr[i]
                    if risk > 0 and profit >= 2.0 * risk:
                        new_signal = SIZE_HALF
                        position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Calculate trailing stop
            current_stop = close[i] + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    profit = entry_price - close[i]
                    risk = 2.0 * atr[int(i)] if i > 0 else atr[i]
                    if risk > 0 and profit >= 2.0 * risk:
                        new_signal = -SIZE_HALF
                        position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals