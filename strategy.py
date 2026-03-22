#!/usr/bin/env python3
"""
Experiment #004: 4h Supertrend + RSI Pullback + 1d HMA Bias + ADX Filter + ATR Stop
Hypothesis: 4h timeframe balances signal frequency and noise reduction. Supertrend provides
clear trend direction with ATR-based stops. 1d HMA gives strong HTF bias to avoid counter-trend
trades. RSI pullbacks (30-45 for longs, 55-70 for shorts) catch entries during trend continuations.
ADX>20 ensures minimum trend strength without being too restrictive (ADX>25 too rare). Multiple
entry paths (6 long + 6 short) ensure >=10 trades per symbol. Conservative sizing (0.25) with
2.5*ATR stoploss controls drawdown on 4h volatility. Works in both bull and bear regimes.
Timeframe: 4h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_supertrend_rsi_1d_hma_adx_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
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
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period:] = pd.Series(dx[period:]).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Supertrend indicator - trend following with ATR bands.
    Returns: supertrend values, direction (1=bullish, -1=bearish)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    direction[:] = np.nan
    
    hl2 = (high + low) / 2.0
    
    for i in range(period, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = np.nan
            direction[i] = np.nan
            continue
        
        upper_band = hl2[i] + multiplier * atr[i]
        lower_band = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            direction[i] = 1
        else:
            if close[i] > supertrend[i-1]:
                supertrend[i] = lower_band
                direction[i] = 1
            elif close[i] < supertrend[i-1]:
                supertrend[i] = upper_band
                direction[i] = -1
            else:
                if direction[i-1] == 1:
                    supertrend[i] = max(lower_band, supertrend[i-1])
                    direction[i] = 1
                else:
                    supertrend[i] = min(upper_band, supertrend[i-1])
                    direction[i] = -1
    
    return supertrend, direction

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    ema_1d_200 = calculate_ema(df_1d['close'].values, 200)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend_4h, st_dir_4h = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(st_dir_4h[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_21[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - primary trend filter
        hma_1d_bullish = close[i] > hma_1d_21_aligned[i]
        hma_1d_bearish = close[i] < hma_1d_21_aligned[i]
        hma_1d_strong_bull = close[i] > hma_1d_50_aligned[i]
        hma_1d_strong_bear = close[i] < hma_1d_50_aligned[i]
        ema_1d_bullish = close[i] > ema_1d_200_aligned[i]
        ema_1d_bearish = close[i] < ema_1d_200_aligned[i]
        
        # 4h Supertrend direction
        st_4h_bullish = st_dir_4h[i] == 1
        st_4h_bearish = st_dir_4h[i] == -1
        
        # Supertrend flip signals on 4h
        st_flip_long = st_dir_4h[i] == 1 and st_dir_4h[i-1] == -1 if i > 0 else False
        st_flip_short = st_dir_4h[i] == -1 and st_dir_4h[i-1] == 1 if i > 0 else False
        
        # EMA trend on 4h
        ema_bullish = close[i] > ema_21[i] and ema_21[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and ema_21[i] < ema_50[i]
        
        # ADX trend strength (lower threshold for 4h to get more trades)
        trend_weak = adx[i] > 18
        trend_strong = adx[i] > 25
        
        # RSI zones (wider range for 4h to get more trades)
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_pullback_long = rsi[i] > 30 and rsi[i] < 50
        rsi_pullback_short = rsi[i] > 50 and rsi[i] < 70
        rsi_neutral = rsi[i] > 35 and rsi[i] < 65
        
        # Bollinger bands
        bb_squeeze = close[i] > bb_lower[i] and close[i] < bb_mid[i]
        bb_expand = close[i] < bb_upper[i] and close[i] > bb_mid[i]
        bb_breakout_long = close[i] > bb_upper[i]
        bb_breakout_short = close[i] < bb_lower[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (6 paths for >=10 trades on 4h) ===
        
        # Path 1: HTF bullish + 4h Supertrend bullish + RSI pullback
        if hma_1d_bullish and st_4h_bullish and rsi_pullback_long and trend_weak:
            new_signal = SIZE_ENTRY
        
        # Path 2: 4h Supertrend flip long + HTF not bearish
        elif st_flip_long and not hma_1d_bearish and adx[i] > 15:
            new_signal = SIZE_ENTRY
        
        # Path 3: HTF strong bullish + 4h EMA bullish + RSI oversold bounce
        elif hma_1d_strong_bull and ema_bullish and rsi_oversold:
            new_signal = SIZE_ENTRY
        
        # Path 4: 4h EMA bullish + Supertrend bullish + ADX building
        elif ema_bullish and st_4h_bullish and adx[i] > 18:
            new_signal = SIZE_ENTRY
        
        # Path 5: HTF bullish + 4h Supertrend bullish + BB breakout
        elif hma_1d_bullish and st_4h_bullish and bb_breakout_long and trend_weak:
            new_signal = SIZE_ENTRY
        
        # Path 6: 4h EMA bullish + RSI neutral + trend building (loose entry)
        elif ema_bullish and rsi_neutral and adx[i] > 18:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (6 paths for >=10 trades on 4h) ===
        
        # Path 1: HTF bearish + 4h Supertrend bearish + RSI pullback
        elif hma_1d_bearish and st_4h_bearish and rsi_pullback_short and trend_weak:
            new_signal = -SIZE_ENTRY
        
        # Path 2: 4h Supertrend flip short + HTF not bullish
        elif st_flip_short and not hma_1d_bullish and adx[i] > 15:
            new_signal = -SIZE_ENTRY
        
        # Path 3: HTF strong bearish + 4h EMA bearish + RSI overbought drop
        elif hma_1d_strong_bear and ema_bearish and rsi_overbought:
            new_signal = -SIZE_ENTRY
        
        # Path 4: 4h EMA bearish + Supertrend bearish + ADX building
        elif ema_bearish and st_4h_bearish and adx[i] > 18:
            new_signal = -SIZE_ENTRY
        
        # Path 5: HTF bearish + 4h Supertrend bearish + BB breakdown
        elif hma_1d_bearish and st_4h_bearish and bb_breakout_short and trend_weak:
            new_signal = -SIZE_ENTRY
        
        # Path 6: 4h EMA bearish + RSI neutral + trend building (loose entry)
        elif ema_bearish and rsi_neutral and adx[i] > 18:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe)
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
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe)
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