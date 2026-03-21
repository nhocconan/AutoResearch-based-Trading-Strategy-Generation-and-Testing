#!/usr/bin/env python3
"""
Experiment #464: 30m HMA Trend + 4h Bias + RSI Pullback + Volume Confirm + ATR Stop
Hypothesis: 30m timeframe offers balance between noise reduction and trade frequency.
Using 4h HMA for trend bias (HTF), 30m RSI(7) for faster pullback detection,
volume spike confirmation to filter false breakouts, and 2.5*ATR stoploss.
Multiple entry paths ensure >=10 trades requirement is met.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_4h_bias_rsi7_volume_atr_v1"
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

def calculate_volume_spike(volume, period=20):
    """Detect volume spikes relative to recent average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_std = vol_s.rolling(window=period, min_periods=period).std().values
    vol_zscore = np.where(vol_std > 0, (volume - vol_avg) / vol_std, 0.0)
    return vol_zscore

def calculate_slope(values, lookback=5):
    """Calculate slope of values over lookback period."""
    n = len(values)
    slope = np.zeros(n)
    for i in range(lookback, n):
        if not np.isnan(values[i]) and not np.isnan(values[i - lookback]):
            slope[i] = (values[i] - values[i - lookback]) / lookback
    return slope

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
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    hma_30m = calculate_hma(close, 21)
    hma_30m_fast = calculate_hma(close, 9)
    rsi = calculate_rsi(close, 7)  # Faster RSI for 30m
    rsi_std = calculate_rsi(close, 14)
    hma_slope = calculate_slope(hma_30m, lookback=5)
    vol_zscore = calculate_volume_spike(volume, 20)
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_30m[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(hma_slope[i]) or np.isnan(vol_zscore[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 30m HMA trend
        hma_30m_bullish = close[i] > hma_30m[i]
        hma_30m_bearish = close[i] < hma_30m[i]
        hma_rising = hma_slope[i] > 0
        hma_falling = hma_slope[i] < 0
        
        # Fast HMA crossover
        fast_above_slow = hma_30m_fast[i] > hma_30m[i]
        fast_below_slow = hma_30m_fast[i] < hma_30m[i]
        
        # RSI zones (faster RSI=7 for 30m)
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral_long = rsi[i] > 40 and rsi[i] < 55
        rsi_neutral_short = rsi[i] > 45 and rsi[i] < 60
        
        # Volume confirmation
        vol_confirmed = vol_zscore[i] > 0.5  # Above average volume
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: 4h bullish + 30m bullish + RSI pullback + HMA rising
        if hma_4h_bullish and hma_30m_bullish and rsi_neutral_long and hma_rising:
            new_signal = SIZE_ENTRY
        # Path 2: 4h bullish + Fast HMA above slow + RSI > 35
        elif hma_4h_bullish and fast_above_slow and rsi[i] > 35 and rsi[i] < 60:
            new_signal = SIZE_ENTRY
        # Path 3: 30m bullish + HMA rising + RSI oversold (deep pullback)
        elif hma_30m_bullish and hma_rising and rsi_oversold:
            new_signal = SIZE_ENTRY
        # Path 4: 4h bullish + 30m bullish + Volume spike + RSI > 40
        elif hma_4h_bullish and hma_30m_bullish and vol_confirmed and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Path 5: Price above both HMA + RSI 40-55 (consolidation breakout)
        elif close[i] > hma_30m[i] and close[i] > hma_4h_aligned[i] and rsi[i] > 40 and rsi[i] < 55:
            new_signal = SIZE_ENTRY
        # Path 6: Fast HMA crossover up + RSI rising + volume
        elif fast_above_slow and rsi[i] > rsi[i-1] and vol_zscore[i] > 0:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: 4h bearish + 30m bearish + RSI pullback + HMA falling
        if hma_4h_bearish and hma_30m_bearish and rsi_neutral_short and hma_falling:
            new_signal = -SIZE_ENTRY
        # Path 2: 4h bearish + Fast HMA below slow + RSI < 60
        elif hma_4h_bearish and fast_below_slow and rsi[i] > 40 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 3: 30m bearish + HMA falling + RSI overbought (rally short)
        elif hma_30m_bearish and hma_falling and rsi_overbought:
            new_signal = -SIZE_ENTRY
        # Path 4: 4h bearish + 30m bearish + Volume spike + RSI < 60
        elif hma_4h_bearish and hma_30m_bearish and vol_confirmed and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 5: Price below both HMA + RSI 45-60 (consolidation breakdown)
        elif close[i] < hma_30m[i] and close[i] < hma_4h_aligned[i] and rsi[i] > 45 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 6: Fast HMA crossover down + RSI falling + volume
        elif fast_below_slow and rsi[i] < rsi[i-1] and vol_zscore[i] > 0:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 30m timeframe)
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
            
            # Calculate trailing stop (2.5*ATR for 30m timeframe)
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