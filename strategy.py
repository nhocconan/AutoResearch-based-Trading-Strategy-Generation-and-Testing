#!/usr/bin/env python3
"""
Experiment #318: 1d KAMA Trend + Weekly HMA Bias + RSI Momentum + ATR Stops
Hypothesis: KAMA adapts to market volatility better than EMA/HMA, reducing whipsaws
in ranging markets while capturing trends efficiently. Weekly HMA provides macro bias.
Wider RSI ranges (25-75) ensure sufficient trade generation (avoiding 0-trade failure).
Simple dual-entry logic (breakout + pullback) maximizes opportunities on 1d timeframe.
Target: Beat Sharpe=0.499 with cleaner trend capture and adequate trade frequency.
Timeframe: 1d (required), HTF: 1w for macro trend bias.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_weekly_hma_rsi_momentum_atr_v1"
timeframe = "1d"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.nan
    volatility = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=er_period, min_periods=er_period).sum().values
    volatility[:er_period] = np.nan
    
    er = np.where(volatility > 0, change / volatility, 0.0)
    er = np.nan_to_num(er, nan=0.0)
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    sma_50 = calculate_sma(close, 50)
    
    # Track previous values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_kama = np.roll(kama, 1)
    prev_kama[0] = kama[0]
    
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
        if np.isnan(kama[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        # Weekly macro trend bias
        weekly_bullish = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        weekly_bearish = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # KAMA trend direction
        kama_bullish = kama[i] > prev_kama[i] and close[i] > kama[i]
        kama_bearish = kama[i] < prev_kama[i] and close[i] < kama[i]
        
        # KAMA crossover signals
        kama_cross_long = close[i] > kama[i] and prev_close[i] <= prev_kama[i]
        kama_cross_short = close[i] < kama[i] and prev_close[i] >= prev_kama[i]
        
        # Price vs SMA50 filter
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # RSI filter (WIDER ranges to ensure trades - avoid 0-trade failure)
        rsi_ok_long = 25 < rsi[i] < 80  # Not extremely overbought
        rsi_ok_short = 20 < rsi[i] < 75  # Not extremely oversold
        rsi_momentum_long = rsi[i] > 45
        rsi_momentum_short = rsi[i] < 55
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Weekly bullish + KAMA bullish + RSI momentum + Above SMA50
        if weekly_bullish and kama_bullish and rsi_momentum_long and above_sma50:
            new_signal = SIZE_ENTRY
        # Secondary: KAMA cross long + Weekly bullish + RSI OK
        elif kama_cross_long and weekly_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Tertiary: KAMA bullish + Above SMA50 + RSI 40-70 (momentum continuation)
        elif kama_bullish and above_sma50 and 40 < rsi[i] < 70:
            new_signal = SIZE_ENTRY
        # Quaternary: Weekly bullish + Above SMA50 + KAMA rising (simple trend)
        elif weekly_bullish and above_sma50 and kama[i] > prev_kama[i]:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Primary: Weekly bearish + KAMA bearish + RSI momentum + Below SMA50
        if weekly_bearish and kama_bearish and rsi_momentum_short and below_sma50:
            new_signal = -SIZE_ENTRY
        # Secondary: KAMA cross short + Weekly bearish + RSI OK
        elif kama_cross_short and weekly_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: KAMA bearish + Below SMA50 + RSI 30-60 (momentum continuation)
        elif kama_bearish and below_sma50 and 30 < rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Quaternary: Weekly bearish + Below SMA50 + KAMA falling (simple trend)
        elif weekly_bearish and below_sma50 and kama[i] < prev_kama[i]:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest)
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
            
            # Calculate trailing stop (2.0*ATR from lowest)
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