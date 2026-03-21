#!/usr/bin/env python3
"""
Experiment #253: 15m Fisher Transform Reversals + 4h HMA Trend Bias
Hypothesis: Fisher Transform excels at 15m timeframe for catching reversals. 
4h HMA provides simple trend bias (price above = bullish bias). 
Entry when Fisher crosses extreme levels WITH trend alignment.
Simpler conditions than failed strategies to ensure sufficient trades.
Position sizing: 0.25 entry, 0.125 half at 2R. Stoploss: 2*ATR trailing.
Target: Beat Sharpe=0.499 with consistent trades across all symbols.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_fisher_4h_hma_rsi_volume_atr_v1"
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
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher(close, high, low, period=9):
    """Calculate Ehlers Fisher Transform for reversal detection."""
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    # Calculate highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    # Normalize price to -1 to +1 range
    normalized = 2 * ((close_s - ll) / (hh - ll + 1e-10)) - 1
    normalized = np.clip(normalized, -0.99, 0.99)
    # Apply Fisher Transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
    fisher = fisher.fillna(0)
    return fisher.values

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

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (0-1, >0.5 = bullish)."""
    ratio = np.where(volume > 0, taker_buy_volume / volume, 0.5)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    fisher = calculate_fisher(close, high, low, 9)
    rsi = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    
    # Track previous values for crossover detection
    prev_fisher = np.roll(fisher, 1)
    prev_fisher[0] = fisher[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filter (simple: price vs 4h HMA)
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # Fisher Transform signals (reversal detection)
        fisher_cross_up = prev_fisher[i] < -1.0 and fisher[i] >= -1.0
        fisher_cross_down = prev_fisher[i] > 1.0 and fisher[i] <= 1.0
        fisher_deep_oversold = fisher[i] < -1.5
        fisher_deep_overbought = fisher[i] > 1.5
        
        # RSI confirmation (looser thresholds for more trades)
        rsi_bullish = rsi[i] > 35
        rsi_bearish = rsi[i] < 65
        rsi_not_extreme = 25 < rsi[i] < 75
        
        # Volume confirmation (optional boost)
        vol_bullish = vol_ratio[i] > 0.50
        vol_bearish = vol_ratio[i] < 0.50
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Fisher cross up from oversold with trend alignment
        if fisher_cross_up:
            if trend_bullish and rsi_bullish:
                new_signal = SIZE_ENTRY
            elif rsi_not_extreme and vol_bullish:
                new_signal = SIZE_ENTRY
        
        # Deep oversold Fisher reversal (strong signal)
        elif fisher_deep_oversold:
            if trend_bullish or (rsi[i] < 40 and vol_bullish):
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Fisher cross down from overbought with trend alignment
        if fisher_cross_down:
            if trend_bearish and rsi_bearish:
                new_signal = -SIZE_ENTRY
            elif rsi_not_extreme and vol_bearish:
                new_signal = -SIZE_ENTRY
        
        # Deep overbought Fisher reversal (strong signal)
        elif fisher_deep_overbought:
            if trend_bearish or (rsi[i] > 60 and vol_bearish):
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2*ATR from highest)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2*ATR from lowest)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals