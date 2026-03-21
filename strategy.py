#!/usr/bin/env python3
"""
Experiment #451: 15m Multi-Timeframe Trend Following with 4h HMA Bias + 1h RSI Pullback
Hypothesis: 15m timeframe captures intraday moves while 4h HMA provides strong trend filter.
1h RSI pullback ensures we enter on dips (not chasing). Volume confirmation reduces false signals.
Multiple entry paths ensure >=10 trades requirement is met. 2*ATR stoploss protects capital.
Timeframe: 15m (REQUIRED), HTF: 1h and 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_4h_hma_1h_rsi_vol_pullback_atr_v1"
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
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    hma_15m = calculate_hma(close, 21)
    hma_15m_fast = calculate_hma(close, 9)
    rsi_15m = calculate_rsi(close, 14)
    hma_slope = calculate_slope(hma_15m, lookback=5)
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
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_15m[i]) or np.isnan(hma_slope[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF - strongest filter)
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h RSI pullback zones (entry timing)
        rsi_1h_oversold = rsi_1h_aligned[i] < 45
        rsi_1h_overbought = rsi_1h_aligned[i] > 55
        rsi_1h_neutral_long = rsi_1h_aligned[i] > 35 and rsi_1h_aligned[i] < 55
        rsi_1h_neutral_short = rsi_1h_aligned[i] > 45 and rsi_1h_aligned[i] < 65
        
        # 15m local trend
        hma_15m_bullish = close[i] > hma_15m[i]
        hma_15m_bearish = close[i] < hma_15m[i]
        hma_rising = hma_slope[i] > 0
        hma_falling = hma_slope[i] < 0
        
        # Fast HMA crossover
        fast_above_slow = hma_15m_fast[i] > hma_15m[i]
        fast_below_slow = hma_15m_fast[i] < hma_15m[i]
        
        # 15m RSI
        rsi_15m_oversold = rsi_15m[i] < 40
        rsi_15m_overbought = rsi_15m[i] > 60
        rsi_15m_neutral = rsi_15m[i] > 35 and rsi_15m[i] < 65
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 0.8
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: 4h bullish + 1h RSI pullback + 15m bullish + volume
        if trend_4h_bullish and rsi_1h_neutral_long and hma_15m_bullish and vol_confirmed:
            new_signal = SIZE_ENTRY
        # Path 2: 4h bullish + 15m HMA rising + RSI oversold
        elif trend_4h_bullish and hma_rising and rsi_15m_oversold:
            new_signal = SIZE_ENTRY
        # Path 3: 4h bullish + Fast HMA above slow + 15m RSI neutral
        elif trend_4h_bullish and fast_above_slow and rsi_15m_neutral:
            new_signal = SIZE_ENTRY
        # Path 4: 4h bullish + 15m bullish + Fast HMA crossover up
        elif trend_4h_bullish and hma_15m_bullish and fast_above_slow and hma_15m_fast[i] > hma_15m_fast[i-1]:
            new_signal = SIZE_ENTRY
        # Path 5: 4h bullish + 1h RSI oversold (deep pullback entry)
        elif trend_4h_bullish and rsi_1h_oversold and vol_confirmed:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: 4h bearish + 1h RSI pullback + 15m bearish + volume
        if trend_4h_bearish and rsi_1h_neutral_short and hma_15m_bearish and vol_confirmed:
            new_signal = -SIZE_ENTRY
        # Path 2: 4h bearish + 15m HMA falling + RSI overbought
        elif trend_4h_bearish and hma_falling and rsi_15m_overbought:
            new_signal = -SIZE_ENTRY
        # Path 3: 4h bearish + Fast HMA below slow + 15m RSI neutral
        elif trend_4h_bearish and fast_below_slow and rsi_15m_neutral:
            new_signal = -SIZE_ENTRY
        # Path 4: 4h bearish + 15m bearish + Fast HMA crossover down
        elif trend_4h_bearish and hma_15m_bearish and fast_below_slow and hma_15m_fast[i] < hma_15m_fast[i-1]:
            new_signal = -SIZE_ENTRY
        # Path 5: 4h bearish + 1h RSI overbought (rally short entry)
        elif trend_4h_bearish and rsi_1h_overbought and vol_confirmed:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2*ATR for 15m timeframe)
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
            
            # Calculate trailing stop (2*ATR for 15m timeframe)
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
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
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