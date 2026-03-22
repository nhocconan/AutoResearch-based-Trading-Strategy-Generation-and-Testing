#!/usr/bin/env python3
"""
Experiment #005: 12h Multi-Timeframe HMA Trend + 1d Bias + RSI Pullback + ATR Stop
Hypothesis: 12h timeframe captures medium-term trends with less noise than lower TFs.
1d HMA provides strong daily trend bias, 12h Supertrend gives entry timing,
12h RSI pullbacks catch entries in direction of HTF trend. ADX filter ensures
minimum trend strength without being too restrictive (>15 not >25). Conservative
sizing (0.30) controls drawdown. 2.5*ATR stoploss appropriate for 12h bars.
Multiple entry paths ensure >=10 trades per symbol even on slower 12h timeframe.
Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_supertrend_1d_rsi_adx_atr_v1"
timeframe = "12h"
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

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    ema_1d_50 = calculate_ema(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend_12h, st_dir_12h = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    hma_12h = calculate_hma(close, 21)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    
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
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(ema_1d_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(st_dir_12h[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - primary trend filter
        hma_1d_bullish = close[i] > hma_1d_aligned[i]
        hma_1d_bearish = close[i] < hma_1d_aligned[i]
        ema_1d_bullish = close[i] > ema_1d_50_aligned[i]
        ema_1d_bearish = close[i] < ema_1d_50_aligned[i]
        
        # Strong HTF trend confirmation
        htf_strong_bull = hma_1d_bullish and ema_1d_bullish
        htf_strong_bear = hma_1d_bearish and ema_1d_bearish
        
        # 12h Supertrend direction
        st_12h_bullish = st_dir_12h[i] == 1
        st_12h_bearish = st_dir_12h[i] == -1
        
        # Supertrend flip signals on 12h
        st_flip_long = st_dir_12h[i] == 1 and st_dir_12h[i-1] == -1 if i > 0 else False
        st_flip_short = st_dir_12h[i] == -1 and st_dir_12h[i-1] == 1 if i > 0 else False
        
        # EMA trend on 12h
        ema_bullish = close[i] > ema_21[i] and ema_21[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and ema_21[i] < ema_50[i]
        
        # HMA trend on 12h
        hma_bullish = close[i] > hma_12h[i]
        hma_bearish = close[i] < hma_12h[i]
        
        # ADX trend strength (lower threshold for 12h to get more trades)
        trend_weak = adx[i] > 15
        trend_strong = adx[i] > 20
        
        # RSI zones (wider range for 12h to get more trades)
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_pullback_long = rsi[i] > 35 and rsi[i] < 55
        rsi_pullback_short = rsi[i] > 45 and rsi[i] < 65
        rsi_neutral = rsi[i] > 35 and rsi[i] < 65
        
        # MACD histogram
        macd_bullish = macd_hist[i] > 0
        macd_bearish = macd_hist[i] < 0
        macd_cross_up = macd_hist[i] > 0 and macd_hist[i-1] <= 0 if i > 0 else False
        macd_cross_down = macd_hist[i] < 0 and macd_hist[i-1] >= 0 if i > 0 else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES (6 paths for >=10 trades on 12h) ===
        
        # Path 1: HTF bullish + 12h Supertrend bullish + RSI pullback
        if htf_strong_bull and st_12h_bullish and rsi_pullback_long and trend_weak:
            new_signal = SIZE_ENTRY
        
        # Path 2: 12h Supertrend flip long + HTF not bearish
        elif st_flip_long and not hma_1d_bearish and adx[i] > 12:
            new_signal = SIZE_ENTRY
        
        # Path 3: HTF bullish + 12h EMA bullish + RSI oversold bounce
        elif htf_strong_bull and ema_bullish and rsi_oversold and rsi[i] > rsi[i-1] if i > 0 else False:
            new_signal = SIZE_ENTRY
        
        # Path 4: 12h HMA bullish + Supertrend bullish + MACD positive
        elif hma_bullish and st_12h_bullish and macd_bullish and trend_weak:
            new_signal = SIZE_ENTRY
        
        # Path 5: HTF bullish + 12h Supertrend bullish + RSI neutral + ADX building
        elif htf_strong_bull and st_12h_bullish and rsi_neutral and adx[i] > adx[i-1] if i > 0 else False and adx[i] > 15:
            new_signal = SIZE_ENTRY
        
        # Path 6: MACD cross up + HTF not bearish + Supertrend bullish
        elif macd_cross_up and not hma_1d_bearish and st_12h_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (6 paths for >=10 trades on 12h) ===
        
        # Path 1: HTF bearish + 12h Supertrend bearish + RSI pullback
        if htf_strong_bear and st_12h_bearish and rsi_pullback_short and trend_weak:
            new_signal = -SIZE_ENTRY
        
        # Path 2: 12h Supertrend flip short + HTF not bullish
        elif st_flip_short and not hma_1d_bullish and adx[i] > 12:
            new_signal = -SIZE_ENTRY
        
        # Path 3: HTF bearish + 12h EMA bearish + RSI overbought drop
        elif htf_strong_bear and ema_bearish and rsi_overbought and rsi[i] < rsi[i-1] if i > 0 else False:
            new_signal = -SIZE_ENTRY
        
        # Path 4: 12h HMA bearish + Supertrend bearish + MACD negative
        elif hma_bearish and st_12h_bearish and macd_bearish and trend_weak:
            new_signal = -SIZE_ENTRY
        
        # Path 5: HTF bearish + 12h Supertrend bearish + RSI neutral + ADX building
        elif htf_strong_bear and st_12h_bearish and rsi_neutral and adx[i] > adx[i-1] if i > 0 else False and adx[i] > 15:
            new_signal = -SIZE_ENTRY
        
        # Path 6: MACD cross down + HTF not bullish + Supertrend bearish
        elif macd_cross_down and not hma_1d_bullish and st_12h_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 12h timeframe - wider stops)
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
            
            # Calculate trailing stop (2.5*ATR for 12h timeframe - wider stops)
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