#!/usr/bin/env python3
"""
Experiment #002: 30m Supertrend + RSI Pullback + 4h HMA Trend Filter
Hypothesis: Supertrend provides clear trend direction with ATR-based stops.
Combined with RSI(14) pullback entries in the direction of 4h trend.
Simpler than CRSI, more reliable trade generation. 4h HMA filters counter-trend trades.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper.
Position sizing: 0.25 base, 0.30 max, discrete levels to minimize fee churn.
Key innovation: Supertrend(10,3) for trend + RSI(14) 30-70 range for pullback entries.
Stoploss: 2.5*ATR trailing stop to limit drawdown in volatile crypto markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_supertrend_rsi_4h_hma_v1"
timeframe = "30m"
leverage = 1.0

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Supertrend indicator - trend following with ATR-based bands.
    Returns: supertrend_values, trend_direction (1=up, -1=down)
    """
    n = len(close)
    atr = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    
    # Calculate ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    # Calculate Supertrend
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend[0] = upper_band[0]
    trend[0] = 1
    
    for i in range(1, n):
        if trend[i-1] == 1:
            if close[i] < supertrend[i-1]:
                trend[i] = -1
                supertrend[i] = upper_band[i]
            else:
                trend[i] = 1
                supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            if close[i] > supertrend[i-1]:
                trend[i] = 1
                supertrend[i] = lower_band[i]
            else:
                trend[i] = -1
                supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    return supertrend, trend

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    if n < period + 1:
        return rsi
    
    gains = np.zeros(n)
    losses = np.zeros(n)
    
    for i in range(1, n):
        change = close[i] - close[i-1]
        if change > 0:
            gains[i] = change
        else:
            losses[i] = -change
    
    avg_gain = np.mean(gains[1:period+1])
    avg_loss = np.mean(losses[1:period+1])
    
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100 - (100 / (1 + rs))
    
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    bandwidth[np.isnan(bandwidth)] = 0.0
    return upper, lower, bandwidth, sma

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
    
    # Calculate 30m indicators
    supertrend, st_trend = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Additional trend filters
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels to minimize fee churn (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.30
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - primary filter
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 30m Supertrend direction
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        # RSI pullback levels
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_extreme_oversold = rsi[i] < 30
        rsi_extreme_overbought = rsi[i] > 70
        
        # Price position vs Bollinger Bands
        price_near_lower = close[i] < bb_lower[i] * 1.01
        price_near_upper = close[i] > bb_upper[i] * 0.99
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i]
        ema_bearish = close[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: 4h bull + 30m Supertrend bull + RSI pullback oversold
        if bull_trend_4h and st_bullish and rsi_oversold:
            new_signal = SIZE_BASE
        # Secondary: 4h bull + RSI extreme oversold + price near lower BB
        elif bull_trend_4h and rsi_extreme_oversold and price_near_lower:
            new_signal = SIZE_MAX
        # Tertiary: 4h bull + Supertrend bull + EMA bullish + RSI < 50
        elif bull_trend_4h and st_bullish and ema_bullish and rsi[i] < 50:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Primary: 4h bear + 30m Supertrend bear + RSI pullback overbought
        if bear_trend_4h and st_bearish and rsi_overbought:
            new_signal = -SIZE_BASE
        # Secondary: 4h bear + RSI extreme overbought + price near upper BB
        elif bear_trend_4h and rsi_extreme_overbought and price_near_upper:
            new_signal = -SIZE_MAX
        # Tertiary: 4h bear + Supertrend bear + EMA bearish + RSI > 50
        elif bear_trend_4h and st_bearish and ema_bearish and rsi[i] > 50:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = int(np.sign(new_signal))
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = int(np.sign(new_signal))
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals