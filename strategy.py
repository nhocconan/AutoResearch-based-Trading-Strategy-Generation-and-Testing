#!/usr/bin/env python3
"""
Experiment #011: 12h KAMA Adaptive Trend + Daily HMA Bias + RSI Entry + ATR Stop
Hypothesis: KAMA adapts to volatility better than EMA/HMA, reducing whipsaws in 2022 crash.
12h timeframe captures multi-day swings. Daily HMA provides HTF bias alignment.
RSI pullback entries (not extremes) ensure >=10 trades per symbol.
Simple logic: fewer filters = more trades. Conservative sizing (0.25) controls DD.
2.5*ATR stoploss appropriate for 12h bars. Must beat Sharpe=0.499 baseline.
Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_daily_hma_rsi_atr_v1"
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
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency ratio.
    Less lag in trends, more smoothing in chop.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.abs(close[:period] - close[0])
    
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-period:i+1])))
    volatility[:period] = change[:period]
    
    er = np.zeros(n)
    er[period:] = np.where(volatility[period:] > 0, change[period:] / volatility[period:], 0)
    
    # Smoothing constant
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # KAMA calculation
    kama[period-1] = close[period-1]
    for i in range(period, n):
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_momentum(close, period=10):
    """Calculate Rate of Change momentum."""
    roc = np.zeros(len(close))
    roc[period:] = (close[period:] - close[:-period]) / close[:-period] * 100
    roc[:period] = np.nan
    return roc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, 10, 2, 30)
    kama_fast = calculate_kama(close, 5, 2, 20)
    rsi = calculate_rsi(close, 14)
    momentum = calculate_momentum(close, 10)
    
    # 12h SMA for additional trend filter
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(momentum[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (HTF) - primary filter
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # 12h KAMA trend
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        kama_rising = kama[i] > kama[i-1] if i > 0 else False
        kama_falling = kama[i] < kama[i-1] if i > 0 else False
        
        # Fast KAMA crossover
        fast_above_slow = kama_fast[i] > kama[i]
        fast_below_slow = kama_fast[i] < kama[i]
        fast_cross_up = kama_fast[i] > kama[i] and kama_fast[i-1] <= kama[i-1] if i > 0 else False
        fast_cross_down = kama_fast[i] < kama[i] and kama_fast[i-1] >= kama[i-1] if i > 0 else False
        
        # Momentum
        mom_positive = momentum[i] > 0
        mom_negative = momentum[i] < 0
        mom_strong = momentum[i] > 2.0
        mom_weak = momentum[i] < -2.0
        
        # RSI zones - pullback entries (not extremes for more trades)
        rsi_pullback_long = rsi[i] > 35 and rsi[i] < 55
        rsi_pullback_short = rsi[i] > 45 and rsi[i] < 65
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_rising = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_falling = rsi[i] < rsi[i-1] if i > 0 else False
        
        # Price position
        price_above_sma50 = close[i] > sma_50[i]
        price_below_sma50 = close[i] < sma_50[i]
        price_above_sma20 = close[i] > sma_20[i]
        price_below_sma20 = close[i] < sma_20[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Daily bullish + KAMA bullish + Fast KAMA cross up + RSI ok
        if daily_bullish and kama_bullish and fast_cross_up and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        
        # Path 2: Daily bullish + KAMA rising + Momentum positive + Price > SMA20
        elif daily_bullish and kama_rising and mom_positive and price_above_sma20:
            new_signal = SIZE_ENTRY
        
        # Path 3: Daily bullish + Fast above Slow KAMA + RSI rising from oversold
        elif daily_bullish and fast_above_slow and rsi_oversold and rsi_rising:
            new_signal = SIZE_ENTRY
        
        # Path 4: Daily bullish + Price > KAMA + Momentum building
        elif daily_bullish and price_above_sma50 and kama_bullish and momentum[i] > momentum[i-1] if i > 0 else False:
            new_signal = SIZE_ENTRY
        
        # Path 5: Simple - Daily bullish + KAMA cross up (price crosses above KAMA)
        elif daily_bullish and close[i] > kama[i] and close[i-1] <= kama[i-1] if i > 0 else False:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Daily bearish + KAMA bearish + Fast KAMA cross down + RSI ok
        if daily_bearish and kama_bearish and fast_cross_down and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        
        # Path 2: Daily bearish + KAMA falling + Momentum negative + Price < SMA20
        elif daily_bearish and kama_falling and mom_negative and price_below_sma20:
            new_signal = -SIZE_ENTRY
        
        # Path 3: Daily bearish + Fast below Slow KAMA + RSI falling from overbought
        elif daily_bearish and fast_below_slow and rsi_overbought and rsi_falling:
            new_signal = -SIZE_ENTRY
        
        # Path 4: Daily bearish + Price < KAMA + Momentum weakening
        elif daily_bearish and price_below_sma50 and kama_bearish and momentum[i] < momentum[i-1] if i > 0 else False:
            new_signal = -SIZE_ENTRY
        
        # Path 5: Simple - Daily bearish + KAMA cross down (price crosses below KAMA)
        elif daily_bearish and close[i] < kama[i] and close[i-1] >= kama[i-1] if i > 0 else False:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 12h timeframe)
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
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 12h timeframe)
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