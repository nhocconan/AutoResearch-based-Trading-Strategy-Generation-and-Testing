#!/usr/bin/env python3
"""
Experiment #188: 30m Donchian Breakout with 4h HMA Trend Filter
Hypothesis: 30m timeframe captures intraday-to-multiday breakouts effectively.
Donchian Channel (20-period) breakout provides clear entry signals that trigger
frequently enough (>10 trades per symbol). 4h HMA(21) filters trend direction
to avoid counter-trend breakouts. RSI(14) filter prevents entries at extremes.
ATR(14) stoploss at 2.5*ATR protects capital. Position sizing 0.25 entry, 0.125
half-size at 2R profit. This targets trending periods (2021 bull, 2024 recovery)
while avoiding whipsaws in 2022 crash and 2025 bear via HTF filter.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_donchian_4h_hma_rsi_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    change = np.abs(close - np.roll(close, er_period))
    change[0:er_period] = np.abs(close[0:er_period] - close[0])
    volatility = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=er_period, min_periods=er_period).sum().values
    volatility[0] = change[0]
    volatility = np.where(volatility > 0, volatility, 1e-10)
    er = change / volatility
    er = np.clip(er, 0, 1)
    
    fast_sc = (2 / (fast_period + 1)) ** 2
    slow_sc = (2 / (slow_period + 1)) ** 2
    sc = er ** 2 * (fast_sc - slow_sc) + slow_sc
    
    kama = np.zeros(len(close))
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
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    hma_20 = calculate_hma(close, 20)
    hma_50 = calculate_hma(close, 50)
    
    # Volume momentum (30m)
    volume = prices["volume"].values
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(volume_sma > 0, volume_sma, 1e-10)
    
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
        # HTF trend filter (4h HMA)
        hma_4h_valid = hma_4h_aligned[i] > 0
        trend_4h_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_4h_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 30m trend
        trend_30m_bullish = hma_20[i] > hma_50[i] and hma_20[i] > kama[i]
        trend_30m_bearish = hma_20[i] < hma_50[i] and hma_20[i] < kama[i]
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # RSI filter (not at extremes)
        rsi_ok_long = 30 < rsi[i] < 70
        rsi_ok_short = 30 < rsi[i] < 70
        
        # Volume confirmation
        volume_ok = volume_ratio[i] > 1.0
        
        # KAMA slope
        kama_rising = kama[i] > kama[i-3] if i > 3 else False
        kama_falling = kama[i] < kama[i-3] if i > 3 else False
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Donchian breakout + 4h bullish + volume + RSI ok
        if breakout_long and trend_4h_bullish and volume_ok and rsi_ok_long:
            new_signal = SIZE_ENTRY
        
        # Secondary: Trend continuation (KAMA rising + 4h bullish + pullback)
        elif trend_30m_bullish and kama_rising and trend_4h_bullish:
            if rsi[i] < 55 and rsi[i] > rsi[i-2] if i > 2 else False:
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Primary: Donchian breakdown + 4h bearish + volume + RSI ok
        if breakout_short and trend_4h_bearish and volume_ok and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        
        # Secondary: Trend continuation (KAMA falling + 4h bearish + bounce)
        elif trend_30m_bearish and kama_falling and trend_4h_bearish:
            if rsi[i] > 45 and rsi[i] < rsi[i-2] if i > 2 else False:
                if new_signal == 0.0:  # Don't override long signal
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