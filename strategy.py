#!/usr/bin/env python3
"""
Experiment #463: 15m KAMA + 4h HMA Trend + 1h RSI Momentum + ATR Stop
Hypothesis: 15m timeframe captures intraday moves while 4h HMA provides trend bias.
KAMA (Kaufman Adaptive Moving Average) adapts to volatility - fast in trends, slow in ranges.
1h RSI confirms momentum direction. Multiple entry paths ensure >=10 trades.
Conservative sizing (0.25) with 2.5*ATR stoploss controls drawdown.
Timeframe: 15m (REQUIRED), HTF: 4h and 1h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_kama_4h_hma_1h_rsi_atr_v1"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    change = np.abs(close - np.roll(close, period))
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1)))
    
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
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
    """Calculate Hull Moving Average."""
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
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, period=10)
    kama_fast = calculate_kama(close, period=5)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 1e10
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(kama_fast[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h momentum (HTF) - wide ranges for more trades
        momentum_long = rsi_1h_aligned[i] > 40 and rsi_1h_aligned[i] < 70
        momentum_short = rsi_1h_aligned[i] > 30 and rsi_1h_aligned[i] < 60
        
        # 15m KAMA crossover
        kama_cross_up = kama_fast[i] > kama[i] and kama_fast[i-1] <= kama[i-1]
        kama_cross_down = kama_fast[i] < kama[i] and kama_fast[i-1] >= kama[i-1]
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # KAMA alignment (simpler than crossover, more trades)
        kama_aligned_long = kama_fast[i] > kama[i]
        kama_aligned_short = kama_fast[i] < kama[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: 4h bullish + 1h momentum + KAMA cross up (highest quality)
        if trend_bullish and momentum_long and kama_cross_up:
            new_signal = SIZE_ENTRY
        # Path 2: 4h bullish + KAMA aligned + 1h RSI 40-60 (more frequent)
        elif trend_bullish and kama_aligned_long and rsi_1h_aligned[i] > 40 and rsi_1h_aligned[i] < 60:
            new_signal = SIZE_ENTRY
        # Path 3: Price above KAMA + 1h RSI 45-60 (consolidation breakout)
        elif price_above_kama and rsi_1h_aligned[i] > 45 and rsi_1h_aligned[i] < 60:
            new_signal = SIZE_ENTRY
        # Path 4: 4h bullish + KAMA cross up (simpler, ensures trades)
        elif trend_bullish and kama_cross_up:
            new_signal = SIZE_ENTRY
        # Path 5: KAMA aligned long + 1h RSI > 45 (momentum continuation)
        elif kama_aligned_long and rsi_1h_aligned[i] > 45 and rsi_1h_aligned[i] < 65:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: 4h bearish + 1h momentum + KAMA cross down (highest quality)
        if trend_bearish and momentum_short and kama_cross_down:
            new_signal = -SIZE_ENTRY
        # Path 2: 4h bearish + KAMA aligned + 1h RSI 40-60 (more frequent)
        elif trend_bearish and kama_aligned_short and rsi_1h_aligned[i] > 40 and rsi_1h_aligned[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 3: Price below KAMA + 1h RSI 40-55 (consolidation breakdown)
        elif price_below_kama and rsi_1h_aligned[i] > 40 and rsi_1h_aligned[i] < 55:
            new_signal = -SIZE_ENTRY
        # Path 4: 4h bearish + KAMA cross down (simpler, ensures trades)
        elif trend_bearish and kama_cross_down:
            new_signal = -SIZE_ENTRY
        # Path 5: KAMA aligned short + 1h RSI < 55 (momentum continuation)
        elif kama_aligned_short and rsi_1h_aligned[i] > 35 and rsi_1h_aligned[i] < 55:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 15m timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 15m timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 1e10
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 1e10
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
            lowest_close = 1e10
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals