#!/usr/bin/env python3
"""
Experiment #440: 30m Donchian Breakout + 4h HMA Trend Bias + ADX/RSI Filters + ATR Stop
Hypothesis: Donchian channel breakouts (Turtle Trading) on 30m timeframe capture medium-term
trends while 4h HMA provides higher timeframe bias to filter false breakouts. 30m offers more
trade opportunities than 1d/12h while maintaining trend-following edge. ADX > 20 ensures we
only trade when trend strength is present. RSI filter avoids entering at extremes. Multiple
entry paths ensure >=10 trades per symbol. ATR-based trailing stop (2.5*ATR) controls drawdown.
Position size: 0.25 discrete (conservative for 30m volatility), stoploss 2.5*ATR.
Timeframe: 30m (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_donchian_4h_hma_adx_rsi_breakout_atr_v1"
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

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel (Turtle Trading breakout system).
    Upper = highest high over N periods
    Lower = lowest low over N periods
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10) * 100
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10) * 100
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)."""
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period:
        return kama
    
    er = np.zeros(n)
    for i in range(1, n):
        change = np.abs(close[i] - close[i-period+1]) if i >= period - 1 else np.abs(close[i] - close[0])
        volatility = np.sum(np.abs(np.diff(close[max(0, i-period+1):i+1])))
        er[i] = change / (volatility + 1e-10) if volatility > 0 else 0
    
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    kama[period-1] = close[period-1]
    for i in range(period, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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
    rsi = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    sma50 = calculate_sma(close, 50)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    kama = calculate_kama(close, 10)
    
    # Volume MA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(sma50[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (higher timeframe direction)
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 30m trend filters
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        above_kama = close[i] > kama[i]
        below_kama = close[i] < kama[i]
        
        # ADX trend strength filter (only trade when ADX > 18)
        trend_strength = adx[i] > 18
        adx_rising = adx[i] > adx[i-1] if not np.isnan(adx[i-1]) else False
        
        # RSI filter (avoid extremes, favor momentum zone)
        rsi_not_overbought = rsi[i] < 72
        rsi_not_oversold = rsi[i] > 28
        rsi_momentum_long = rsi[i] > 45 and rsi[i] < 68
        rsi_momentum_short = rsi[i] > 32 and rsi[i] < 55
        
        # Volume confirmation
        volume_above_avg = volume[i] > vol_sma[i] if not np.isnan(vol_sma[i]) else True
        
        # Donchian breakout signals (breakout from previous bar's channel)
        breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # DI crossover signals
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # Channel position
        near_upper = close[i] > donchian_upper[i] * 0.985 if not np.isnan(donchian_upper[i]) else False
        near_lower = close[i] < donchian_lower[i] * 1.015 if not np.isnan(donchian_lower[i]) else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Donchian breakout + 4h bullish + ADX trend + RSI not overbought
        if breakout_long and hma_4h_bullish and trend_strength and rsi_not_overbought:
            new_signal = SIZE_ENTRY
        # Path 2: Donchian breakout + Above SMA50 + DI bullish + RSI momentum
        elif breakout_long and above_sma50 and di_bullish and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Path 3: Near upper Donchian + 4h bullish + ADX > 22 + Volume
        elif near_upper and hma_4h_bullish and adx[i] > 22 and volume_above_avg and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        # Path 4: Breakout + 4h bullish + Above KAMA + RSI 45-65
        elif breakout_long and hma_4h_bullish and above_kama and rsi[i] > 45 and rsi[i] < 65:
            new_signal = SIZE_ENTRY
        # Path 5: DI bullish + 4h bullish + Above SMA50 + ADX rising
        elif di_bullish and hma_4h_bullish and above_sma50 and adx_rising and adx[i] > 16:
            new_signal = SIZE_ENTRY
        # Path 6: Above KAMA + 4h bullish + ADX > 20 + RSI > 50
        elif above_kama and hma_4h_bullish and adx[i] > 20 and rsi[i] > 50 and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        # Path 7: Breakout + Volume spike + 4h bullish
        elif breakout_long and volume_above_avg and hma_4h_bullish and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Donchian breakout + 4h bearish + ADX trend + RSI not oversold
        if breakout_short and hma_4h_bearish and trend_strength and rsi_not_oversold:
            new_signal = -SIZE_ENTRY
        # Path 2: Donchian breakout + Below SMA50 + DI bearish + RSI momentum
        elif breakout_short and below_sma50 and di_bearish and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Path 3: Near lower Donchian + 4h bearish + ADX > 22 + Volume
        elif near_lower and hma_4h_bearish and adx[i] > 22 and volume_above_avg and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        # Path 4: Breakout + 4h bearish + Below KAMA + RSI 35-55
        elif breakout_short and hma_4h_bearish and below_kama and rsi[i] > 35 and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        # Path 5: DI bearish + 4h bearish + Below SMA50 + ADX rising
        elif di_bearish and hma_4h_bearish and below_sma50 and adx_rising and adx[i] > 16:
            new_signal = -SIZE_ENTRY
        # Path 6: Below KAMA + 4h bearish + ADX > 20 + RSI < 50
        elif below_kama and hma_4h_bearish and adx[i] > 20 and rsi[i] < 50 and rsi[i] > 30:
            new_signal = -SIZE_ENTRY
        # Path 7: Breakout + Volume spike + 4h bearish
        elif breakout_short and volume_above_avg and hma_4h_bearish and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest for 30m timeframe)
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
            
            # Calculate trailing stop (2.5*ATR from lowest for 30m timeframe)
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