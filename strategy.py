#!/usr/bin/env python3
"""
Experiment #158: 30m RSI Pullback with 4h HMA Trend Filter and ATR Stoploss
Hypothesis: 30m timeframe captures intermediate swings better than 1h/4h for mean-reversion
within trend. Using 4h HMA for major trend bias + 30m RSI pullback entries generates
sufficient trades while avoiding counter-trend traps. Looser RSI thresholds (35-45 long,
55-65 short) ensure trades trigger on all symbols. ATR stoploss at 2.0*ATR protects capital.
This addresses the 0-trade failure mode by simplifying entry conditions while maintaining
trend alignment via 4h HTF filter. Position sizing: 0.25 entry, 0.125 at 2R profit.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_pullback_4h_hma_atr_v1"
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

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs recent average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = np.where(vol_avg > 0, volume / vol_avg, 1.0)
    return vol_ratio

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
    rsi = calculate_rsi(close, 14)
    sma_50 = calculate_sma(close, 50)
    hma_20 = calculate_hma(close, 20)
    hma_50 = calculate_hma(close, 50)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
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
        # 4h trend filter (major trend direction)
        hma_4h_valid = hma_4h_aligned[i] > 0
        trend_4h_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_4h_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 30m trend filter
        trend_30m_bullish = hma_20[i] > hma_50[i] if hma_50[i] > 0 else False
        trend_30m_bearish = hma_20[i] < hma_50[i] if hma_50[i] > 0 else False
        
        # Price position relative to SMA50
        above_sma50 = sma_50[i] > 0 and close[i] > sma_50[i]
        below_sma50 = sma_50[i] > 0 and close[i] < sma_50[i]
        
        # RSI pullback signals (LOOSE thresholds for more trades)
        rsi_pullback_long = 35 <= rsi[i] <= 50
        rsi_pullback_short = 50 <= rsi[i] <= 65
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # Volume confirmation
        volume_ok = vol_ratio[i] >= 0.7  # At least 70% of avg volume
        
        # RSI momentum
        rsi_rising = rsi[i] > rsi[i-1] if i > 1 else False
        rsi_falling = rsi[i] < rsi[i-1] if i > 1 else False
        
        new_signal = 0.0
        
        # LONG ENTRY: 4h bullish + RSI pullback OR 4h neutral + RSI oversold
        if rsi_pullback_long or rsi_oversold:
            if trend_4h_bullish and volume_ok:
                # Strong long: 4h trend + pullback
                new_signal = SIZE_ENTRY
            elif not trend_4h_bearish and rsi_rising and above_sma50:
                # Moderate long: 4h not bearish + RSI turning up
                new_signal = SIZE_ENTRY
            elif trend_30m_bullish and rsi_oversold and volume_ok:
                # 30m trend + oversold
                new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: 4h bearish + RSI pullback OR 4h neutral + RSI overbought
        if new_signal == 0.0 and (rsi_pullback_short or rsi_overbought):
            if trend_4h_bearish and volume_ok:
                # Strong short: 4h trend + pullback
                new_signal = -SIZE_ENTRY
            elif not trend_4h_bullish and rsi_falling and below_sma50:
                # Moderate short: 4h not bullish + RSI turning down
                new_signal = -SIZE_ENTRY
            elif trend_30m_bearish and rsi_overbought and volume_ok:
                # 30m trend + overbought
                new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from lowest)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
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
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
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