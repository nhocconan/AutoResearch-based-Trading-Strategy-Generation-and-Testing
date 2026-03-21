#!/usr/bin/env python3
"""
Experiment #296: 30m KAMA Trend + 4h HMA Macro + RSI Pullback with ATR Stops

Hypothesis: 30-minute timeframe captures intraday momentum swings while 4-hour HMA provides 
macro trend bias. KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than 
simple EMA/HMA, reducing whipsaws in ranging markets. RSI pullback entries (35-55 long, 
45-65 short) ensure we enter on dips in uptrends. ATR trailing stops (2.5*ATR) control 
drawdown. Position size 0.28 balances returns vs risk.

This should generate MORE trades than 1h/4h/12h strategies due to faster timeframe,
while 4h trend filter prevents counter-trend trades that failed in previous experiments.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_4h_hma_rsi_pullback_atr_v1"
timeframe = "30m"
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
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if i < period:
            kama[i] = close[i]
            continue
        
        change = np.abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[max(0, i-period):i+1])))
        
        if volatility > 0:
            er = change / volatility
        else:
            er = 0.0
        
        sc = (er * (2.0/(fast+1) - 2.0/(slow+1)) + 2.0/(slow+1)) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    kama_10 = calculate_kama(close, 10)
    kama_30 = calculate_kama(close, 30)
    rsi = calculate_rsi(close, 14)
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Track previous values for crossover detection
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_kama_10 = np.roll(kama_10, 1)
    prev_kama_10[0] = kama_10[0]
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(kama_10[i]) or np.isnan(kama_30[i]) or np.isnan(atr[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # 4h macro trend bias
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 30m KAMA trend
        kama_bullish = kama_10[i] > kama_30[i]
        kama_bearish = kama_10[i] < kama_30[i]
        
        # KAMA slope
        kama_slope_bullish = kama_10[i] > prev_kama_10[i]
        kama_slope_bearish = kama_10[i] < prev_kama_10[i]
        
        # RSI pullback zones (generous ranges to ensure trades)
        rsi_pullback_long = 30 < rsi[i] < 55
        rsi_pullback_short = 45 < rsi[i] < 70
        rsi_not_extreme_long = rsi[i] < 75
        rsi_not_extreme_short = rsi[i] > 25
        
        # Volume confirmation (above average)
        volume_confirmed = volume[i] > vol_ma[i] * 0.8
        
        # KAMA crossover signals
        kama_cross_long = prev_kama_10[i] <= kama_30[i] and kama_10[i] > kama_30[i]
        kama_cross_short = prev_kama_10[i] >= kama_30[i] and kama_10[i] < kama_30[i]
        
        # Price position relative to KAMA
        price_above_kama = close[i] > kama_10[i]
        price_below_kama = close[i] < kama_10[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        if trend_4h_bullish and kama_bullish and rsi_pullback_long and kama_cross_long:
            new_signal = SIZE_ENTRY
        elif trend_4h_bullish and kama_slope_bullish and rsi_pullback_long and price_above_kama:
            new_signal = SIZE_ENTRY
        elif kama_bullish and rsi_not_extreme_long and kama_cross_long and volume_confirmed:
            new_signal = SIZE_ENTRY
        elif price_above_kama and kama_10[i] > kama_30[i] and 40 < rsi[i] < 60 and kama_slope_bullish:
            new_signal = SIZE_ENTRY
        elif trend_4h_bullish and price_above_kama and rsi[i] > 40 and rsi_not_extreme_long:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        if trend_4h_bearish and kama_bearish and rsi_pullback_short and kama_cross_short:
            new_signal = -SIZE_ENTRY
        elif trend_4h_bearish and kama_slope_bearish and rsi_pullback_short and price_below_kama:
            new_signal = -SIZE_ENTRY
        elif kama_bearish and rsi_not_extreme_short and kama_cross_short and volume_confirmed:
            new_signal = -SIZE_ENTRY
        elif price_below_kama and kama_10[i] < kama_30[i] and 40 < rsi[i] < 60 and kama_slope_bearish:
            new_signal = -SIZE_ENTRY
        elif trend_4h_bearish and price_below_kama and rsi[i] < 60 and rsi_not_extreme_short:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals