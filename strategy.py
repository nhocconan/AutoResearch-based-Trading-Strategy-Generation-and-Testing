#!/usr/bin/env python3
"""
Experiment #005: 12h Donchian Breakout + Daily KAMA Bias + RSI Filter + ATR Stop
Hypothesis: Donchian breakouts work well on higher timeframes (12h) where noise is reduced.
Daily KAMA provides adaptive HTF trend bias (KAMA adapts to volatility better than HMA/EMA).
RSI filter ensures we enter on momentum confirmation, not just price breaks.
Multiple entry paths (breakout + pullback + momentum) ensure >=10 trades per symbol.
12h timeframe = fewer false signals than lower TFs, but still enough trades vs 1d.
Conservative sizing (0.28) with 2.5*ATR stoploss controls drawdown in 2022 crash.
Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_daily_kama_rsi_atr_v1"
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
    Adapts to market volatility - moves fast in trends, slow in ranges.
    Better than EMA/HMA for regime changes.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period + slow:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        price_change = np.abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[max(0, i-period):i+1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc = sc ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - tracks highest high and lowest low over period.
    Breakout above upper = long signal, below lower = short signal.
    """
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

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

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram for momentum confirmation."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    kama_1d = calculate_kama(df_1d['close'].values, period=10, fast=2, slow=30)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    rsi = calculate_rsi(close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    
    # 12h KAMA for local trend
    kama_12h = calculate_kama(close, period=10, fast=2, slow=30)
    kama_12h_fast = calculate_kama(close, period=5, fast=2, slow=15)
    
    # 12h SMA for additional filter
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
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
    
    for i in range(250, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_1d_aligned[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(macd_hist[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (HTF) - KAMA adaptive
        daily_bullish = close[i] > kama_1d_aligned[i]
        daily_bearish = close[i] < kama_1d_aligned[i]
        daily_kama_rising = kama_1d_aligned[i] > kama_1d_aligned[i-1] if i > 0 else False
        daily_kama_falling = kama_1d_aligned[i] < kama_1d_aligned[i-1] if i > 0 else False
        
        # 12h Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # 12h KAMA trend
        kama_12h_bullish = close[i] > kama_12h[i]
        kama_12h_bearish = close[i] < kama_12h[i]
        kama_12h_rising = kama_12h[i] > kama_12h[i-1] if i > 0 else False
        kama_12h_falling = kama_12h[i] < kama_12h[i-1] if i > 0 else False
        
        # Fast KAMA crossover
        fast_above_slow = kama_12h_fast[i] > kama_12h[i]
        fast_below_slow = kama_12h_fast[i] < kama_12h[i]
        
        # MACD momentum
        macd_bullish = macd_hist[i] > 0
        macd_bearish = macd_hist[i] < 0
        macd_cross_up = macd_hist[i] > 0 and macd_hist[i-1] <= 0 if i > 0 else False
        macd_cross_down = macd_hist[i] < 0 and macd_hist[i-1] >= 0 if i > 0 else False
        
        # RSI zones
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        rsi_momentum_long = rsi[i] > 45 and rsi[i] < 65
        rsi_momentum_short = rsi[i] > 35 and rsi[i] < 55
        
        # SMA trend filter
        above_sma50 = close[i] > sma_50[i] if not np.isnan(sma_50[i]) else False
        below_sma50 = close[i] < sma_50[i] if not np.isnan(sma_50[i]) else False
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Donchian breakout + Daily bullish + RSI momentum + MACD bullish
        if breakout_long and daily_bullish and rsi_momentum_long and macd_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 2: Daily bullish + KAMA 12h bullish + Fast KAMA crossover up + RSI > 50
        elif daily_bullish and kama_12h_bullish and fast_above_slow and rsi_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 3: MACD cross up + Daily not bearish + RSI rising + Above SMA50
        elif macd_cross_up and not daily_bearish and rsi[i] > rsi[i-1] if i > 0 else False and above_sma50:
            new_signal = SIZE_ENTRY
        
        # Path 4: Daily KAMA rising + Price pullback to KAMA 12h + RSI oversold bounce
        elif daily_kama_rising and close[i] < kama_12h[i] * 1.02 and close[i] > kama_12h[i] * 0.98 and rsi_oversold and rsi[i] > rsi[i-1] if i > 0 else False:
            new_signal = SIZE_ENTRY
        
        # Path 5: Above SMA200 + Donchian near breakout + MACD improving + RSI neutral
        elif above_sma200 and close[i] > donchian_upper[i-1] * 0.98 if i > 0 else False and macd_hist[i] > macd_hist[i-1] if i > 0 else False and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: Donchian breakout + Daily bearish + RSI momentum + MACD bearish
        if breakout_short and daily_bearish and rsi_momentum_short and macd_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 2: Daily bearish + KAMA 12h bearish + Fast KAMA crossover down + RSI < 50
        elif daily_bearish and kama_12h_bearish and fast_below_slow and rsi_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 3: MACD cross down + Daily not bullish + RSI falling + Below SMA50
        elif macd_cross_down and not daily_bullish and rsi[i] < rsi[i-1] if i > 0 else False and below_sma50:
            new_signal = -SIZE_ENTRY
        
        # Path 4: Daily KAMA falling + Price rally to KAMA 12h + RSI overbought drop
        elif daily_kama_falling and close[i] < kama_12h[i] * 1.02 and close[i] > kama_12h[i] * 0.98 and rsi_overbought and rsi[i] < rsi[i-1] if i > 0 else False:
            new_signal = -SIZE_ENTRY
        
        # Path 5: Below SMA200 + Donchian near breakout + MACD worsening + RSI neutral
        elif below_sma200 and close[i] < donchian_lower[i-1] * 1.02 if i > 0 else False and macd_hist[i] < macd_hist[i-1] if i > 0 else False and rsi[i] < 55:
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