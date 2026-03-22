#!/usr/bin/env python3
"""
Experiment #004: 4h Multi-Timeframe Donchian Breakout + 1d HMA Bias + RSI Filter + ATR Stop
Hypothesis: 4h timeframe captures medium-term trends with fewer whipsaws than lower TFs.
Daily HMA provides strong directional bias (only trade in HTF trend direction).
Donchian(20) breakout on 4h captures momentum moves. RSI(14) filter avoids exhausted breakouts.
ADX(14) > 20 ensures trend strength. 2.5*ATR stoploss appropriate for 4h volatility.
Conservative sizing (0.28) controls drawdown during 2022 crash. 1d HTF alignment via mtf_data.
Multiple entry paths ensure >=10 trades per symbol even in ranging markets.
Timeframe: 4h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_1d_hma_rsi_adx_atr_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_kama(close, period=10):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
    
    # Smoothing constant
    sc = (er * (2.0 / (period + 1) - 2.0 / (period + 1)) + 2.0 / (period + 1)) ** 2
    
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

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
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    kama = calculate_kama(close, 10)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
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
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - strongest filter
        hma_1d_bullish = close[i] > hma_1d_aligned[i]
        hma_1d_bearish = close[i] < hma_1d_aligned[i]
        ema_1d_bullish = hma_1d_aligned[i] > ema_1d_50_aligned[i]
        ema_1d_bearish = hma_1d_aligned[i] < ema_1d_50_aligned[i]
        
        # 4h trend indicators
        ema_bullish = close[i] > ema_21[i] and ema_21[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and ema_21[i] < ema_50[i]
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # Donchian breakout signals
        donchian_breakout_long = close[i] > donchian_upper[i - 1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i - 1] if i > 0 else False
        
        # ADX trend strength
        trend_strong = adx[i] > 20
        trend_very_strong = adx[i] > 25
        
        # RSI zones - avoid exhausted moves
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral_long = rsi[i] > 35 and rsi[i] < 60
        rsi_neutral_short = rsi[i] > 40 and rsi[i] < 65
        rsi_momentum_long = rsi[i] > 50 and rsi[i] < 70
        rsi_momentum_short = rsi[i] > 30 and rsi[i] < 50
        
        # RSI momentum (rising/falling)
        rsi_rising = rsi[i] > rsi[i - 1] if i > 0 else False
        rsi_falling = rsi[i] < rsi[i - 1] if i > 0 else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES (4 paths for >=10 trades) ===
        
        # Path 1: 1d HMA bullish + Donchian breakout + RSI momentum + ADX strong
        if hma_1d_bullish and ema_1d_bullish and donchian_breakout_long and rsi_momentum_long and trend_strong:
            new_signal = SIZE_ENTRY
        
        # Path 2: 1d HMA bullish + EMA bullish + KAMA bullish + RSI neutral (pullback entry)
        elif hma_1d_bullish and ema_bullish and kama_bullish and rsi_neutral_long and rsi_rising:
            new_signal = SIZE_ENTRY
        
        # Path 3: 1d HMA bullish + close > EMA21 + RSI oversold bounce (dip buy)
        elif hma_1d_bullish and close[i] > ema_21[i] and rsi_oversold and rsi_rising:
            new_signal = SIZE_ENTRY
        
        # Path 4: 1d HMA bullish + Donchian near breakout + ADX building (early entry)
        elif hma_1d_bullish and close[i] > donchian_upper[i] * 0.98 and adx[i] > adx[i - 1] if i > 0 else False and adx[i] > 18:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (4 paths for >=10 trades) ===
        
        # Path 1: 1d HMA bearish + Donchian breakdown + RSI momentum + ADX strong
        if hma_1d_bearish and ema_1d_bearish and donchian_breakout_short and rsi_momentum_short and trend_strong:
            new_signal = -SIZE_ENTRY
        
        # Path 2: 1d HMA bearish + EMA bearish + KAMA bearish + RSI neutral (pullback entry)
        elif hma_1d_bearish and ema_bearish and kama_bearish and rsi_neutral_short and rsi_falling:
            new_signal = -SIZE_ENTRY
        
        # Path 3: 1d HMA bearish + close < EMA21 + RSI overbought drop (rally short)
        elif hma_1d_bearish and close[i] < ema_21[i] and rsi_overbought and rsi_falling:
            new_signal = -SIZE_ENTRY
        
        # Path 4: 1d HMA bearish + Donchian near breakdown + ADX building (early entry)
        elif hma_1d_bearish and close[i] < donchian_lower[i] * 1.02 and adx[i] > adx[i - 1] if i > 0 else False and adx[i] > 18:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe - wider stops)
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
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe - wider stops)
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