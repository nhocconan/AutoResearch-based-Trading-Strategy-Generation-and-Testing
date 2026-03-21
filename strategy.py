#!/usr/bin/env python3
"""
Experiment #093: 1h Regime-Adaptive with 4h HMA Trend Filter
Hypothesis: Current best uses 12h Supertrend. Try 1h timeframe with regime-adaptive logic:
- Use ADX to detect trending (ADX>25) vs ranging (ADX<20) markets
- Trending regime: EMA crossover + HTF bias + momentum confirmation
- Ranging regime: RSI mean reversion + HTF bias only
- 4h HMA provides proven HTF trend filter (from current best strategy)
- This should generate MORE trades than pure trend-following while maintaining edge
- Position sizing: 0.25 entry, stoploss at 2.5*ATR trailing
- Looser entry conditions to ensure 10+ trades per symbol (learning from 0-trade failures)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_adaptive_4h_hma_v1"
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

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

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

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return sma.values, upper.values, lower.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # EMA for crossover
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    # MACD
    macd_line, signal_line, histogram = calculate_macd(close, 12, 26, 9)
    
    # Bollinger Bands for regime detection
    bb_sma, bb_upper, bb_lower = calculate_bollinger(close, 20, 2.0)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # 4h trend filter (HTF)
        daily_bullish = close[i] > hma_4h_aligned[i]
        daily_bearish = close[i] < hma_4h_aligned[i]
        
        # Regime detection: trending vs ranging
        trend_regime = adx[i] > 22  # ADX > 22 = trending
        range_regime = adx[i] < 18  # ADX < 18 = ranging
        
        # 1h EMA crossover
        ema_cross_long = ema_21[i] > ema_50[i] and (i > 0 and ema_21[i-1] <= ema_50[i-1])
        ema_cross_short = ema_21[i] < ema_50[i] and (i > 0 and ema_21[i-1] >= ema_50[i-1])
        
        # EMA trend state
        ema_trend_long = ema_21[i] > ema_50[i]
        ema_trend_short = ema_21[i] < ema_50[i]
        
        # MACD signals
        macd_bullish = histogram[i] > 0
        macd_bearish = histogram[i] < 0
        macd_cross_long = histogram[i] > 0 and (i > 0 and histogram[i-1] <= 0)
        macd_cross_short = histogram[i] < 0 and (i > 0 and histogram[i-1] >= 0)
        
        # RSI signals
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Directional Movement
        dm_long = plus_di[i] > minus_di[i]
        dm_short = minus_di[i] > plus_di[i]
        
        # Bollinger position
        bb_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if (bb_upper[i] - bb_lower[i]) > 0 else 0.5
        bb_low = bb_position < 0.2
        bb_high = bb_position > 0.8
        
        new_signal = 0.0
        
        # TRENDING REGIME entries (more aggressive, follow momentum)
        if trend_regime:
            # Long: EMA cross + HTF bullish + MACD bullish OR EMA trend + DM long
            if (ema_cross_long and daily_bullish and macd_bullish):
                new_signal = SIZE_ENTRY
            elif (ema_trend_long and daily_bullish and dm_long and rsi_bullish):
                new_signal = SIZE_ENTRY
            elif (macd_cross_long and daily_bullish and ema_trend_long):
                new_signal = SIZE_ENTRY
            
            # Short: EMA cross + HTF bearish + MACD bearish OR EMA trend + DM short
            if (ema_cross_short and daily_bearish and macd_bearish):
                new_signal = -SIZE_ENTRY
            elif (ema_trend_short and daily_bearish and dm_short and rsi_bearish):
                new_signal = -SIZE_ENTRY
            elif (macd_cross_short and daily_bearish and ema_trend_short):
                new_signal = -SIZE_ENTRY
        
        # RANGING REGIME entries (mean reversion, lighter position)
        elif range_regime:
            # Long: RSI oversold + HTF bullish + price near BB lower
            if rsi_oversold and daily_bullish and bb_low:
                new_signal = SIZE_ENTRY * 0.8  # Smaller size in range
            # Short: RSI overbought + HTF bearish + price near BB upper
            elif rsi_overbought and daily_bearish and bb_high:
                new_signal = -SIZE_ENTRY * 0.8
        
        # NEUTRAL REGIME (ADX 18-22): use simpler trend following
        else:
            if ema_trend_long and daily_bullish:
                new_signal = SIZE_ENTRY * 0.7
            elif ema_trend_short and daily_bearish:
                new_signal = -SIZE_ENTRY * 0.7
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
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
                # Take profit at 1.5R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 1.5 * risk:
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
                # Take profit at 1.5R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 1.5 * risk:
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