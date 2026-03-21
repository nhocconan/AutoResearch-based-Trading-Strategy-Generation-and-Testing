#!/usr/bin/env python3
"""
Experiment #385: 15m RSI Mean Reversion + 4h HMA Trend + ATR Stop
Hypothesis: 15m timeframe is too noisy for pure trend following (see #373, #379 failures).
Instead, use RSI mean reversion WITH HTF trend filter. Long when RSI<30 + 4h bullish,
short when RSI>70 + 4h bearish. This captures pullbacks in the direction of the higher
timeframe trend. Loose RSI thresholds (25-75) ensure minimum trade frequency (critical -
many 15m strategies failed with 0 trades). 4h HMA provides trend bias via mtf_data helper.
ATR(14) stoploss at 2.5x protects capital. Position size 0.25 discrete to minimize fees.
Timeframe: 15m (REQUIRED), HTF: 4h for trend bias via mtf_data helper (call ONCE before loop).
Target: Beat Sharpe=0.499 (current best mtf_12h_supertrend_daily_hma_rsi_pullback_v2).
Key insight: 15m mean reversion + 4h trend filter = fewer whipsaws than pure 15m trend following.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_mr_4h_hma_trend_atr_v1"
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
    """Calculate Hull Moving Average for faster trend response with less lag."""
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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

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
    rsi = calculate_rsi(close, 14)
    sma200 = calculate_sma(close, 200)
    
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
    
    for i in range(250, n):  # Start after 250 bars for SMA200 + indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(sma200[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # SMA200 filter for longer-term trend
        above_sma200 = close[i] > sma200[i]
        below_sma200 = close[i] < sma200[i]
        
        # RSI mean reversion signals (LOOSE thresholds for trade frequency)
        rsi_oversold = rsi[i] < 35  # Long entry zone
        rsi_overbought = rsi[i] > 65  # Short entry zone
        rsi_extreme_long = rsi[i] < 25  # Strong long signal
        rsi_extreme_short = rsi[i] > 75  # Strong short signal
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple conditions to ensure trades) ===
        # Primary: RSI oversold + 4h bullish + above SMA200
        if rsi_oversold and trend_bullish and above_sma200:
            new_signal = SIZE_ENTRY
        # Secondary: RSI extreme long + 4h bullish (SMA200 neutral ok)
        elif rsi_extreme_long and trend_bullish:
            new_signal = SIZE_ENTRY
        # Tertiary: RSI oversold + above SMA200 (4h neutral ok)
        elif rsi_oversold and above_sma200:
            new_signal = SIZE_ENTRY
        # Quaternary: RSI extreme long alone (ensures minimum trade frequency)
        elif rsi_extreme_long:
            new_signal = SIZE_ENTRY
        # Quintenary: RSI < 40 + 4h bullish (very loose backup)
        elif rsi[i] < 40 and trend_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple conditions to ensure trades) ===
        # Primary: RSI overbought + 4h bearish + below SMA200
        if rsi_overbought and trend_bearish and below_sma200:
            new_signal = -SIZE_ENTRY
        # Secondary: RSI extreme short + 4h bearish (SMA200 neutral ok)
        elif rsi_extreme_short and trend_bearish:
            new_signal = -SIZE_ENTRY
        # Tertiary: RSI overbought + below SMA200 (4h neutral ok)
        elif rsi_overbought and below_sma200:
            new_signal = -SIZE_ENTRY
        # Quaternary: RSI extreme short alone (ensures minimum trade frequency)
        elif rsi_extreme_short:
            new_signal = -SIZE_ENTRY
        # Quintenary: RSI > 60 + 4h bearish (very loose backup)
        elif rsi[i] > 60 and trend_bearish:
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