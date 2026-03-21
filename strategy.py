#!/usr/bin/env python3
"""
Experiment #319: 15m Multi-Timeframe RSI Pullback + 4h HMA Trend + 1h Momentum Filter
Hypothesis: 15m primary timeframe captures intraday momentum while 4h HMA provides 
macro trend bias. 1h RSI pullback entries reduce whipsaws. Multiple entry conditions
ensure sufficient trade frequency. ATR trailing stops protect capital during reversals.
Timeframe: 15m (required), HTF: 4h for trend, 1h for momentum confirmation.
Target: Beat Sharpe=0.499 with more frequent entries and better trend alignment.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_4h_hma_1h_rsi_pullback_multi_entry_atr_v1"
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
    """Calculate Hull Moving Average for faster trend response."""
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

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average for trend filter."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
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
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 4h HMA for macro trend
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h RSI for momentum
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 1h HMA for intermediate trend
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_15m = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    
    # Track previous values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_rsi = np.roll(rsi_15m, 1)
    prev_rsi[0] = rsi_15m[0]
    prev_macd_hist = np.roll(macd_hist, 1)
    prev_macd_hist[0] = macd_hist[0]
    
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
        if np.isnan(atr[i]) or np.isnan(rsi_15m[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # === TREND BIAS (4h HMA) ===
        hma_4h_valid = not np.isnan(hma_4h_aligned[i])
        trend_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # === INTERMEDIATE TREND (1h HMA) ===
        hma_1h_valid = not np.isnan(hma_1h_aligned[i])
        hma_1h_bullish = hma_1h_valid and close[i] > hma_1h_aligned[i]
        hma_1h_bearish = hma_1h_valid and close[i] < hma_1h_aligned[i]
        
        # === 1h RSI MOMENTUM ===
        rsi_1h_valid = not np.isnan(rsi_1h_aligned[i])
        rsi_1h_bullish = rsi_1h_valid and 40 < rsi_1h_aligned[i] < 70
        rsi_1h_bearish = rsi_1h_valid and 30 < rsi_1h_aligned[i] < 60
        
        # === 15m RSI PULLBACK ===
        rsi_pullback_long = 35 < rsi_15m[i] < 55 and prev_rsi[i] <= 35
        rsi_pullback_short = 45 < rsi_15m[i] < 65 and prev_rsi[i] >= 65
        rsi_oversold = rsi_15m[i] < 35
        rsi_overbought = rsi_15m[i] > 65
        
        # === EMA CROSSOVER ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        ema_cross_long = ema_21[i] > ema_50[i] and ema_21[i-1] <= ema_50[i-1]
        ema_cross_short = ema_21[i] < ema_50[i] and ema_21[i-1] >= ema_50[i-1]
        
        # === MACD MOMENTUM ===
        macd_bullish = macd_hist[i] > 0
        macd_bearish = macd_hist[i] < 0
        macd_cross_long = macd_hist[i] > 0 and prev_macd_hist[i] <= 0
        macd_cross_short = macd_hist[i] < 0 and prev_macd_hist[i] >= 0
        
        # === SMA 200 FILTER ===
        sma_200_valid = not np.isnan(sma_200[i])
        above_sma200 = sma_200_valid and close[i] > sma_200[i]
        below_sma200 = sma_200_valid and close[i] < sma_200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY (multiple conditions - any can trigger) ===
        # Primary: 4h bullish + 1h bullish + RSI pullback
        if trend_bullish and hma_1h_bullish and rsi_pullback_long:
            new_signal = SIZE_ENTRY
        # Secondary: 4h bullish + EMA bullish + MACD cross long
        elif trend_bullish and ema_bullish and macd_cross_long:
            new_signal = SIZE_ENTRY
        # Tertiary: 4h bullish + above SMA200 + RSI oversold (mean reversion)
        elif trend_bullish and above_sma200 and rsi_oversold:
            new_signal = SIZE_ENTRY
        # Quaternary: 1h bullish + EMA cross long + MACD bullish
        elif hma_1h_bullish and ema_cross_long and macd_bullish:
            new_signal = SIZE_ENTRY
        # Fifth: 4h bullish + 1h RSI bullish + EMA bullish (simple trend)
        elif trend_bullish and rsi_1h_bullish and ema_bullish:
            new_signal = SIZE_ENTRY
        # Sixth: MACD cross long + RSI pullback + above SMA200
        elif macd_cross_long and rsi_pullback_long and above_sma200:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY (multiple conditions - any can trigger) ===
        # Primary: 4h bearish + 1h bearish + RSI pullback
        if trend_bearish and hma_1h_bearish and rsi_pullback_short:
            new_signal = -SIZE_ENTRY
        # Secondary: 4h bearish + EMA bearish + MACD cross short
        elif trend_bearish and ema_bearish and macd_cross_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: 4h bearish + below SMA200 + RSI overbought (mean reversion)
        elif trend_bearish and below_sma200 and rsi_overbought:
            new_signal = -SIZE_ENTRY
        # Quaternary: 1h bearish + EMA cross short + MACD bearish
        elif hma_1h_bearish and ema_cross_short and macd_bearish:
            new_signal = -SIZE_ENTRY
        # Fifth: 4h bearish + 1h RSI bearish + EMA bearish (simple trend)
        elif trend_bearish and rsi_1h_bearish and ema_bearish:
            new_signal = -SIZE_ENTRY
        # Sixth: MACD cross short + RSI pullback + below SMA200
        elif macd_cross_short and rsi_pullback_short and below_sma200:
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