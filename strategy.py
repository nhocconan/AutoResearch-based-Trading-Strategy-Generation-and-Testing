#!/usr/bin/env python3
"""
Experiment #453: 1h KAMA Adaptive Trend + 4h HMA Bias + Volume Confirmation + ATR Stop
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise better than EMA/HMA.
In trending markets, KAMA moves fast; in ranging markets, it flattens (reducing whipsaws).
Combined with 4h HMA bias filter and volume confirmation, this should work better in
bear/range markets (2025+) while capturing trends in bull markets (2021-2024).
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
Position sizing: 0.25 discrete, stoploss at 2.5*ATR, take profit at 2R.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_4h_hma_volume_rsi_atr_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio (ER).
    ER = |net change| / sum of absolute changes over period
    High ER = trending (KAMA moves fast), Low ER = ranging (KAMA flattens)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        net_change = np.abs(close[i] - close[i - er_period])
        sum_changes = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if sum_changes > 0:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0.0
    
    # Calculate smoothing constants
    fast_sc = (2.0 / (fast_period + 1)) ** 2
    slow_sc = (2.0 / (slow_period + 1)) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama, er

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
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

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    kama_1h, er_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Calculate KAMA slope
    kama_slope = np.zeros(n)
    kama_slope[:] = np.nan
    for i in range(5, n):
        if not np.isnan(kama_1h[i]) and not np.isnan(kama_1h[i - 5]):
            kama_slope[i] = (kama_1h[i] - kama_1h[i - 5]) / 5
    
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
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(kama_1h[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(kama_slope[i]) or np.isnan(er_1h[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h KAMA trend
        kama_bullish = close[i] > kama_1h[i]
        kama_bearish = close[i] < kama_1h[i]
        kama_rising = kama_slope[i] > 0
        kama_falling = kama_slope[i] < 0
        
        # Efficiency Ratio (trending vs ranging)
        high_er = er_1h[i] > 0.5  # Strong trend
        low_er = er_1h[i] < 0.3   # Ranging
        
        # Volume confirmation
        volume_confirmed = vol_ratio[i] > 1.2  # 20% above average
        
        # RSI zones
        rsi_bullish_zone = rsi[i] > 40 and rsi[i] < 60  # Neutral-bullish
        rsi_bearish_zone = rsi[i] > 40 and rsi[i] < 60  # Neutral-bearish
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: 4h bullish + KAMA bullish + KAMA rising + RSI zone
        if trend_4h_bullish and kama_bullish and kama_rising and rsi_bullish_zone:
            new_signal = SIZE_ENTRY
        # Path 2: 4h bullish + KAMA bullish + Volume spike + ER high (trending)
        elif trend_4h_bullish and kama_bullish and volume_confirmed and high_er:
            new_signal = SIZE_ENTRY
        # Path 3: 4h bullish + KAMA rising + RSI oversold (pullback entry)
        elif trend_4h_bullish and kama_rising and rsi_oversold:
            new_signal = SIZE_ENTRY
        # Path 4: Price above KAMA + KAMA rising + Volume confirmed
        elif kama_bullish and kama_rising and volume_confirmed:
            new_signal = SIZE_ENTRY
        # Path 5: 4h bullish + Price > KAMA + RSI > 45 (momentum continuation)
        elif trend_4h_bullish and kama_bullish and rsi[i] > 45 and rsi[i] < 65:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: 4h bearish + KAMA bearish + KAMA falling + RSI zone
        if trend_4h_bearish and kama_bearish and kama_falling and rsi_bearish_zone:
            new_signal = -SIZE_ENTRY
        # Path 2: 4h bearish + KAMA bearish + Volume spike + ER high (trending)
        elif trend_4h_bearish and kama_bearish and volume_confirmed and high_er:
            new_signal = -SIZE_ENTRY
        # Path 3: 4h bearish + KAMA falling + RSI overbought (rally short)
        elif trend_4h_bearish and kama_falling and rsi_overbought:
            new_signal = -SIZE_ENTRY
        # Path 4: Price below KAMA + KAMA falling + Volume confirmed
        elif kama_bearish and kama_falling and volume_confirmed:
            new_signal = -SIZE_ENTRY
        # Path 5: 4h bearish + Price < KAMA + RSI < 55 (momentum continuation)
        elif trend_4h_bearish and kama_bearish and rsi[i] > 35 and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 1h timeframe)
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
            
            # Calculate trailing stop (2.5*ATR for 1h timeframe)
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