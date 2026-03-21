#!/usr/bin/env python3
"""
Experiment #107: 12h KAMA Trend + RSI Pullback + Weekly HMA + Volume Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise better than EMA,
providing cleaner trend signals in choppy markets. Combine with RSI pullback entries
(proven in current best), add Weekly HMA as stronger HTF trend filter, and use
taker buy/sell volume ratio for entry confirmation. This should work across all
regimes (bull/bear/range) while generating sufficient trades (20-40/year).
Position sizing: 0.25 entry, 0.125 at 1.5R profit, stoploss at 2.5*ATR trailing.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_rsi_weekly_hma_volume_v1"
timeframe = "12h"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = np.abs(close_s - close_s.shift(period))
    volatility = np.abs(close_s - close_s.shift(1)).rolling(window=period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing Constant
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # KAMA calculation
    kama = pd.Series(index=close_s.index, dtype=float)
    kama.iloc[period-1] = close_s.iloc[period-1]
    
    for i in range(period, len(close_s)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama.iloc[i-1])
    
    return kama.values

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
    """Calculate taker buy volume ratio."""
    ratio = taker_buy_volume / np.maximum(volume, 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama_10 = calculate_kama(close, 10, 2, 30)
    kama_30 = calculate_kama(close, 30, 2, 30)
    volume_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    
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
        # Weekly trend filter (strongest HTF)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Daily trend filter (secondary HTF)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # 12h KAMA trend
        kama_trend_long = kama_10[i] > kama_30[i]
        kama_trend_short = kama_10[i] < kama_30[i]
        
        # KAMA crossover signals
        kama_cross_long = kama_10[i] > kama_30[i] and (i > 0 and kama_10[i-1] <= kama_30[i-1])
        kama_cross_short = kama_10[i] < kama_30[i] and (i > 0 and kama_10[i-1] >= kama_30[i-1])
        
        # RSI pullback signals (buy dips in uptrend, sell rallies in downtrend)
        rsi_pullback_long = rsi[i] < 50 and rsi[i] > 30  # Pullback but not oversold
        rsi_pullback_short = rsi[i] > 50 and rsi[i] < 70  # Rally but not overbought
        
        # Volume confirmation (buying/selling pressure)
        volume_bullish = volume_ratio[i] > 0.55  # More buying pressure
        volume_bearish = volume_ratio[i] < 0.45  # More selling pressure
        
        new_signal = 0.0
        
        # LONG ENTRY conditions (simpler to ensure trades)
        # Condition 1: KAMA cross + Weekly bullish + Volume confirmation
        if kama_cross_long and weekly_bullish and volume_bullish:
            new_signal = SIZE_ENTRY
        # Condition 2: KAMA trend + Daily bullish + RSI pullback + Volume
        elif kama_trend_long and daily_bullish and rsi_pullback_long and volume_bullish:
            new_signal = SIZE_ENTRY
        # Condition 3: KAMA trend + Weekly bullish + RSI not extreme
        elif kama_trend_long and weekly_bullish and rsi[i] > 35 and rsi[i] < 65:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY conditions
        # Condition 1: KAMA cross + Weekly bearish + Volume confirmation
        if kama_cross_short and weekly_bearish and volume_bearish:
            new_signal = -SIZE_ENTRY
        # Condition 2: KAMA trend + Daily bearish + RSI pullback + Volume
        elif kama_trend_short and daily_bearish and rsi_pullback_short and volume_bearish:
            new_signal = -SIZE_ENTRY
        # Condition 3: KAMA trend + Weekly bearish + RSI not extreme
        elif kama_trend_short and weekly_bearish and rsi[i] > 35 and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 1.5R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 1.5 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 1.5R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 1.5 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
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