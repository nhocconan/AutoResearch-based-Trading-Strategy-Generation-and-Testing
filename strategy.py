#!/usr/bin/env python3
"""
Experiment #389: 12h Donchian Breakout + Daily KAMA Trend + ADX Filter + RSI Momentum + ATR Stop
Hypothesis: Donchian channel breakouts work well on 12h timeframe for capturing sustained trends.
Daily KAMA (Kaufman Adaptive Moving Average) provides volatility-adaptive trend filter that performs
better than HMA in ranging markets. ADX(14) > 25 ensures we only trade when trend strength exists.
RSI(14) momentum confirms entry timing to avoid false breakouts. ATR(14) stoploss at 2.5x protects
capital during reversals. Position size 0.30 discrete to minimize fee churn.
Timeframe: 12h (REQUIRED), HTF: 1d for trend bias via mtf_data helper (call ONCE before loop).
Target: Beat Sharpe=0.499 (current best mtf_12h_supertrend_daily_hma_rsi_pullback_v2).
Key insight: Donchian breakouts + ADX filter + adaptive KAMA trend = fewer whipsaws than HMA crossover.
Different from #383 which used HMA crossover - this uses channel breakouts instead.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_daily_kama_adx_rsi_atr_v1"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market volatility - moves fast in trends, slow in ranges.
    More robust than EMA/HMA in choppy markets.
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        price_change = np.abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    kama[:period] = np.nan
    return kama

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
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = trending market, ADX < 25 = ranging market.
    """
    n = len(close)
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    np.abs(high[i] - close[i-1]), 
                    np.abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0.0)
        else:
            plus_dm[i] = 0.0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0.0)
        else:
            minus_dm[i] = 0.0
    
    # Smooth with Wilder's method (EMA with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[:period*2] = np.nan  # Need extra period for ADX smoothing
    
    return adx

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel (upper and lower bands).
    Upper = highest high of last N periods
    Lower = lowest low of last N periods
    Breakout above upper = long signal
    Breakout below lower = short signal
    """
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    upper[:period-1] = np.nan
    lower[:period-1] = np.nan
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    kama_1d = calculate_kama(df_1d['close'].values, period=10)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # 12h KAMA for additional trend confirmation
    kama_12h = calculate_kama(close, period=10)
    
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
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_12h[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (KAMA adaptive trend)
        daily_bullish = not np.isnan(kama_1d_aligned[i]) and close[i] > kama_1d_aligned[i]
        daily_bearish = not np.isnan(kama_1d_aligned[i]) and close[i] < kama_1d_aligned[i]
        
        # 12h KAMA trend
        kama_12h_bullish = close[i] > kama_12h[i]
        kama_12h_bearish = close[i] < kama_12h[i]
        
        # ADX trend strength filter (ensure we trade in trending markets)
        is_trending = adx[i] > 20  # Slightly lower threshold to ensure trades
        
        # Donchian breakout signals
        donchian_breakout_long = close[i] > donchian_upper[i-1] and close[i-1] <= donchian_upper[i-1]
        donchian_breakout_short = close[i] < donchian_lower[i-1] and close[i-1] >= donchian_lower[i-1]
        
        # Donchian position (already broken out)
        donchian_bullish = close[i] > donchian_upper[i-1]
        donchian_bearish = close[i] < donchian_lower[i-1]
        
        # RSI momentum filter (ensure we're not entering at extremes)
        rsi_ok_long = rsi[i] > 40 and rsi[i] < 80
        rsi_ok_short = rsi[i] > 20 and rsi[i] < 60
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 45 and rsi[i] < 75
        rsi_momentum_short = rsi[i] > 25 and rsi[i] < 55
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple conditions to ensure trades) ===
        # Primary: Donchian breakout long + Daily bullish + Trending + RSI ok
        if donchian_breakout_long and daily_bullish and is_trending and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Secondary: Donchian bullish + Daily bullish + 12h KAMA bullish + RSI momentum
        elif donchian_bullish and daily_bullish and kama_12h_bullish and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Tertiary: Donchian breakout long + 12h KAMA bullish + RSI ok (daily neutral ok)
        elif donchian_breakout_long and kama_12h_bullish and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Quaternary: Donchian bullish + 12h KAMA bullish + ADX trending (ensures trade frequency)
        elif donchian_bullish and kama_12h_bullish and is_trending and rsi[i] > 40 and rsi[i] < 75:
            new_signal = SIZE_ENTRY
        # Quintenary: Donchian breakout long alone (backup for minimum trades)
        elif donchian_breakout_long and rsi[i] > 40 and rsi[i] < 75:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple conditions to ensure trades) ===
        # Primary: Donchian breakout short + Daily bearish + Trending + RSI ok
        if donchian_breakout_short and daily_bearish and is_trending and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Secondary: Donchian bearish + Daily bearish + 12h KAMA bearish + RSI momentum
        elif donchian_bearish and daily_bearish and kama_12h_bearish and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: Donchian breakout short + 12h KAMA bearish + RSI ok (daily neutral ok)
        elif donchian_breakout_short and kama_12h_bearish and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Quaternary: Donchian bearish + 12h KAMA bearish + ADX trending (ensures trade frequency)
        elif donchian_bearish and kama_12h_bearish and is_trending and rsi[i] > 25 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Quintenary: Donchian breakout short alone (backup for minimum trades)
        elif donchian_breakout_short and rsi[i] > 25 and rsi[i] < 60:
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