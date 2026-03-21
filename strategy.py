#!/usr/bin/env python3
"""
Experiment #292: 4h KAMA Adaptive Trend with Daily HMA + Choppiness Regime Filter
Hypothesis: KAMA adapts to market noise better than HMA/EMA, reducing whipsaws in 4h timeframe.
Choppiness Index filters range vs trend regimes - only trade breakouts in trending markets (CHOP<38.2).
Daily HMA provides macro trend bias. Wider Donchian(50) reduces false breakouts vs Donchian(20).
Volume confirmation ensures breakout has participation. Conservative sizing (0.25) controls DD.
Target: Beat Sharpe=0.499 from current best while ensuring >=10 trades per symbol.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_daily_hma_chop_donchian_volume_atr_v1"
timeframe = "4h"
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

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average - adapts to market noise."""
    close_s = pd.Series(close)
    change = np.abs(close_s - close_s.shift(er_period))
    volatility = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    er = change / volatility
    er = er.fillna(0)
    
    fast_sc = 2 / (fast_sc + 1)
    slow_sc = 2 / (slow_sc + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index - identifies range vs trend regimes."""
    atr = calculate_atr(high, low, close, period)
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10((highest_high - lowest_low) / atr_sum) / np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=50):
    """Calculate Donchian Channel (upper/lower bounds) - wider period for fewer false breakouts."""
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

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average for volume confirmation."""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    chop = calculate_choppiness(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 50)
    vol_ma = calculate_volume_ma(volume, 20)
    
    # Track previous values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_kama = np.roll(kama, 1)
    prev_kama[0] = kama[0]
    prev_rsi = np.roll(rsi, 1)
    prev_rsi[0] = rsi[0]
    
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
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # HTF trend filter
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # Regime filter - Choppiness Index
        trending_regime = chop[i] < 45  # <38.2 = strong trend, <45 = moderate trend
        ranging_regime = chop[i] > 55   # >61.8 = strong range, >55 = moderate range
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.2 * vol_ma[i]
        
        # KAMA trend direction
        kama_bullish = kama[i] > prev_kama[i]
        kama_bearish = kama[i] < prev_kama[i]
        
        # KAMA crossover signals
        kama_cross_long = kama[i] > prev_kama[i] and prev_close[i] <= prev_kama[i] and close[i] > kama[i]
        kama_cross_short = kama[i] < prev_kama[i] and prev_close[i] >= prev_kama[i] and close[i] < kama[i]
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i-1] and prev_close[i] <= donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1] and prev_close[i] >= donchian_lower[i-1]
        
        # RSI filter
        rsi_ok_long = 35 < rsi[i] < 70
        rsi_ok_short = 30 < rsi[i] < 65
        rsi_not_extreme_long = rsi[i] < 75
        rsi_not_extreme_short = rsi[i] > 25
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # KAMA crossover + Daily HMA bullish + trending regime
        if kama_cross_long and daily_bullish and trending_regime:
            new_signal = SIZE_ENTRY
        # Donchian breakout + Daily HMA bullish + volume confirmed
        elif breakout_long and daily_bullish and volume_confirmed and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # KAMA trend + Daily bullish + RSI pullback
        elif kama_bullish and daily_bullish and 40 < rsi[i] < 55 and close[i] > kama[i]:
            new_signal = SIZE_ENTRY
        # Breakout in trending regime (simpler condition for more trades)
        elif breakout_long and trending_regime and rsi_not_extreme_long:
            new_signal = SIZE_ENTRY
        # KAMA above price + Daily bullish (trend continuation)
        elif kama_bullish and daily_bullish and close[i] > kama[i] and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # KAMA crossover + Daily HMA bearish + trending regime
        if kama_cross_short and daily_bearish and trending_regime:
            new_signal = -SIZE_ENTRY
        # Donchian breakout + Daily HMA bearish + volume confirmed
        elif breakout_short and daily_bearish and volume_confirmed and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # KAMA trend + Daily bearish + RSI pullback
        elif kama_bearish and daily_bearish and 45 < rsi[i] < 60 and close[i] < kama[i]:
            new_signal = -SIZE_ENTRY
        # Breakout in trending regime (simpler condition for more trades)
        elif breakout_short and trending_regime and rsi_not_extreme_short:
            new_signal = -SIZE_ENTRY
        # KAMA below price + Daily bearish (trend continuation)
        elif kama_bearish and daily_bearish and close[i] < kama[i] and rsi[i] < 55:
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