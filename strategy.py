#!/usr/bin/env python3
"""
Experiment #421: 15m Supertrend + 4h HMA Trend + RSI Momentum + ATR Stop
Hypothesis: 15m timeframe needs simpler entry logic than daily strategies. Using 4h HMA
for trend bias (HTF), 15m Supertrend for entries, and relaxed RSI thresholds to ensure
>=10 trades per symbol. Key difference from failed 15m strategies: fewer conflicting filters,
more permissive RSI ranges (35-65 instead of tight extremes), Supertrend as primary trigger.
Position size: 0.25 discrete, stoploss 2.0*ATR for 15m timeframe (tighter than daily).
Timeframe: 15m (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_4h_hma_rsi_momentum_atr_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator.
    Returns: supertrend_line, direction (1=bullish, -1=bearish)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    direction[:] = np.nan
    
    hl2 = (high + low) / 2.0
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend[0] = upper_band[0]
    direction[0] = -1
    
    for i in range(1, n):
        if np.isnan(atr[i]):
            supertrend[i] = np.nan
            direction[i] = np.nan
            continue
        
        # Update bands
        if upper_band[i] < supertrend[i-1] or close[i-1] > supertrend[i-1]:
            supertrend[i] = upper_band[i]
        else:
            supertrend[i] = supertrend[i-1]
        
        if lower_band[i] > supertrend[i-1] or close[i-1] < supertrend[i-1]:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = supertrend[i-1]
        
        # Determine direction
        if close[i] > supertrend[i]:
            direction[i] = 1
            supertrend[i] = lower_band[i]
        else:
            direction[i] = -1
            supertrend[i] = upper_band[i]
    
    return supertrend, direction

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

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = pd.Series(100 * plus_dm / np.where(atr > 0, atr, 1)).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(100 * minus_dm / np.where(atr > 0, atr, 1)).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    sma50 = calculate_sma(close, 50)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
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
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(sma50[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(st_direction[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF direction)
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # 15m Supertrend direction
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # Supertrend flip signals (entry triggers)
        st_flip_bull = st_direction[i] == 1 and st_direction[i-1] == -1 if i > 0 else False
        st_flip_bear = st_direction[i] == -1 and st_direction[i-1] == 1 if i > 0 else False
        
        # RSI momentum (RELAXED for more trades)
        rsi_ok_long = rsi[i] > 35 and rsi[i] < 75
        rsi_ok_short = rsi[i] > 25 and rsi[i] < 65
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 45
        rsi_momentum_short = rsi[i] < 55
        
        # ADX trend strength (avoid choppy markets)
        adx_strong = adx[i] > 20  # Relaxed from 25
        
        # Price vs SMA50
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Supertrend flip + 4h bullish + RSI ok (primary)
        if st_flip_bull and trend_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Path 2: Supertrend bullish + 4h bullish + RSI momentum + ADX strong
        elif st_bullish and trend_bullish and rsi_momentum_long and adx_strong:
            new_signal = SIZE_ENTRY
        # Path 3: Supertrend flip + above SMA50 + RSI ok (4h neutral ok)
        elif st_flip_bull and above_sma50 and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Path 4: 4h bullish + Supertrend bullish + above SMA50 + RSI > 50
        elif trend_bullish and st_bullish and above_sma50 and rsi[i] > 50:
            new_signal = SIZE_ENTRY
        # Path 5: Simple - Supertrend flip + RSI momentum (most permissive)
        elif st_flip_bull and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Supertrend flip + 4h bearish + RSI ok (primary)
        if st_flip_bear and trend_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Path 2: Supertrend bearish + 4h bearish + RSI momentum + ADX strong
        elif st_bearish and trend_bearish and rsi_momentum_short and adx_strong:
            new_signal = -SIZE_ENTRY
        # Path 3: Supertrend flip + below SMA50 + RSI ok (4h neutral ok)
        elif st_flip_bear and below_sma50 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 4: 4h bearish + Supertrend bearish + below SMA50 + RSI < 50
        elif trend_bearish and st_bearish and below_sma50 and rsi[i] < 50:
            new_signal = -SIZE_ENTRY
        # Path 5: Simple - Supertrend flip + RSI momentum (most permissive)
        elif st_flip_bear and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest for 15m timeframe)
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
            
            # Calculate trailing stop (2.0*ATR from lowest for 15m timeframe)
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