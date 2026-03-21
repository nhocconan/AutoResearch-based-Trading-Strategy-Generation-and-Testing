#!/usr/bin/env python3
"""
Experiment #129: 1h KAMA Trend with 4h/12h HMA Filter + RSI Pullback
Hypothesis: 1h timeframe offers better trade frequency than 12h while maintaining
signal quality. Using 4h HMA as primary trend filter (more responsive than daily)
and 12h HMA as secondary confirmation. KAMA adapts to market efficiency, reducing
whipsaws in ranging conditions. Multiple entry paths ensure sufficient trade count.
RSI pullback (30-70 range) captures entries during trend continuation.
Volume confirmation via taker_buy_ratio filters institutional participation.
Position sizing: 0.25 entry, 0.15 at 2R profit, stoploss at 2.5*ATR trailing.
This adapts the near-successful #125 (Sharpe=-0.019) to 1h timeframe with
tighter filters to achieve positive Sharpe across all symbols.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_4h_12h_hma_rsi_volume_v1"
timeframe = "1h"
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
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts smoothing based on market efficiency ratio.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    net_change = close_s.diff(period).abs()
    abs_changes = close_s.diff().abs()
    sum_abs_changes = abs_changes.rolling(window=period, min_periods=period).sum()
    
    er = net_change / sum_abs_changes.replace(0, np.nan)
    er = er.fillna(0)
    
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

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

def calculate_taker_buy_ratio(volume, taker_buy_volume):
    """Calculate taker buy volume ratio."""
    ratio = np.zeros(len(volume))
    mask = volume > 0
    ratio[mask] = taker_buy_volume[mask] / volume[mask]
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
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama_fast = calculate_kama(close, period=10, fast=2, slow=30)
    kama_slow = calculate_kama(close, period=20, fast=2, slow=30)
    taker_ratio = calculate_taker_buy_ratio(volume, taker_buy_volume)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filters (4h and 12h HMA)
        hma4h_bullish = close[i] > hma_4h_aligned[i]
        hma4h_bearish = close[i] < hma_4h_aligned[i]
        hma12h_bullish = close[i] > hma_12h_aligned[i]
        hma12h_bearish = close[i] < hma_12h_aligned[i]
        
        # KAMA trend (fast vs slow)
        kama_trend_long = kama_fast[i] > kama_slow[i]
        kama_trend_short = kama_fast[i] < kama_slow[i]
        
        # KAMA slope (direction)
        kama_slope_long = kama_fast[i] > kama_fast[i-1] if i > 0 else False
        kama_slope_short = kama_fast[i] < kama_fast[i-1] if i > 0 else False
        
        # RSI pullback zones (wider range for more trades on 1h)
        rsi_pullback_long = 30 <= rsi[i] <= 70
        rsi_pullback_short = 30 <= rsi[i] <= 70
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 40
        rsi_momentum_short = rsi[i] < 60
        
        # Volume confirmation
        volume_bullish = taker_ratio[i] > 0.50
        volume_bearish = taker_ratio[i] < 0.50
        
        new_signal = 0.0
        
        # LONG ENTRY conditions (multiple paths for trade frequency)
        # Path 1: KAMA trend + 4h bullish + RSI ok
        if kama_trend_long and kama_slope_long and hma4h_bullish and rsi_pullback_long and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Path 2: KAMA trend + 12h bullish + Volume
        elif kama_trend_long and hma12h_bullish and volume_bullish and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Path 3: KAMA cross + 4h bullish (simpler entry)
        elif kama_trend_long and kama_fast[i-1] <= kama_slow[i-1] and hma4h_bullish:
            new_signal = SIZE_ENTRY
        # Path 4: Both HTF bullish + KAMA alignment
        elif hma12h_bullish and hma4h_bullish and kama_trend_long and rsi[i] > 35:
            new_signal = SIZE_ENTRY
        # Path 5: Strong momentum entry
        elif kama_trend_long and kama_slope_long and rsi[i] > 50 and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY conditions
        # Path 1: KAMA trend + 4h bearish + RSI ok
        if kama_trend_short and kama_slope_short and hma4h_bearish and rsi_pullback_short and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Path 2: KAMA trend + 12h bearish + Volume
        elif kama_trend_short and hma12h_bearish and volume_bearish and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Path 3: KAMA cross + 4h bearish (simpler entry)
        elif kama_trend_short and kama_fast[i-1] >= kama_slow[i-1] and hma4h_bearish:
            new_signal = -SIZE_ENTRY
        # Path 4: Both HTF bearish + KAMA alignment
        elif hma12h_bearish and hma4h_bearish and kama_trend_short and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        # Path 5: Strong momentum entry
        elif kama_trend_short and kama_slope_short and rsi[i] < 50 and rsi[i] > 30:
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