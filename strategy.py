#!/usr/bin/env python3
"""
Experiment #288: 1d HMA Trend + Weekly Macro Filter + RSI Pullback + Donchian Breakout
Hypothesis: Daily timeframe with weekly trend filter reduces whipsaws. RSI pullback entries
in trend direction catch continuations. Donchian breakout confirms momentum. ATR trailing
stop protects capital. Conservative sizing (0.30) limits drawdown during 2022 crash.
Target: Beat Sharpe=0.499 from current best (mtf_12h_supertrend_daily_hma_rsi_pullback_v2)
Position sizing: 0.30 entry, 0.15 half at 2R profit. Stoploss: 2.5*ATR trailing.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_weekly_rsi_donchian_atr_v1"
timeframe = "1d"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    change = np.abs(close - np.roll(close, period))
    change[0] = np.abs(close[0] - close[0])
    volatility = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=period, min_periods=period).sum().values
    volatility[0] = change[0]
    er = np.where(volatility > 0, change / volatility, 0.0)
    sc = (er * (2.0/(fast+1) - 2.0/(slow+1)) + 2.0/(slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    hma_1d = calculate_hma(close, 21)
    kama_1d = calculate_kama(close, 10)
    
    # Track previous values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # HTF trend filters
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # 1d trend filters
        daily_bullish = close[i] > hma_1d[i] and hma_1d[i] > kama_1d[i]
        daily_bearish = close[i] < hma_1d[i] and hma_1d[i] < kama_1d[i]
        
        # RSI pullback zones (not extreme)
        rsi_pullback_long = 35 < rsi[i] < 55
        rsi_pullback_short = 45 < rsi[i] < 65
        rsi_neutral_long = 40 < rsi[i] < 60
        rsi_neutral_short = 40 < rsi[i] < 60
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i-1] and prev_close[i] <= donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1] and prev_close[i] >= donchian_lower[i-1]
        
        # Price above/below Donchian mid
        donchian_mid = (donchian_upper[i-1] + donchian_lower[i-1]) / 2
        above_mid = close[i] > donchian_mid
        below_mid = close[i] < donchian_mid
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Weekly bullish + Daily bullish + RSI pullback
        if weekly_bullish and daily_bullish and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        # Secondary: Weekly bullish + Donchian breakout + RSI neutral
        elif weekly_bullish and breakout_long and rsi_neutral_long:
            new_signal = SIZE_ENTRY
        # Tertiary: Daily bullish + Price above Donchian mid + RSI ok
        elif daily_bullish and above_mid and 35 < rsi[i] < 65:
            new_signal = SIZE_ENTRY
        # Momentum: Weekly bullish + breakout + daily trend
        elif weekly_bullish and breakout_long and daily_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Primary: Weekly bearish + Daily bearish + RSI pullback
        if weekly_bearish and daily_bearish and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        # Secondary: Weekly bearish + Donchian breakout + RSI neutral
        elif weekly_bearish and breakout_short and rsi_neutral_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: Daily bearish + Price below Donchian mid + RSI ok
        elif daily_bearish and below_mid and 35 < rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        # Momentum: Weekly bearish + breakout + daily trend
        elif weekly_bearish and breakout_short and daily_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
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
            
            # Calculate trailing stop (2.5*ATR from lowest)
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