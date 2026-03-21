#!/usr/bin/env python3
"""
Experiment #454: 4h Ehlers Fisher Transform + Daily HMA Trend + ATR Stop
Hypothesis: Fisher Transform excels at catching reversals in bear/range markets (2025 test).
Combined with daily HMA trend filter to avoid counter-trend trades. 4h timeframe balances
trade frequency vs noise. Fisher crosses at extremes (-1.5/+1.5) signal entries.
ATR stoploss (2.5x) protects capital. Multiple entry paths ensure >=10 trades.
Timeframe: 4h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_daily_hma_trend_atr_v1"
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

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals at extremes. Period=9 is standard.
    """
    close_s = pd.Series(close)
    # Calculate highest high and lowest low over period
    hh = close_s.rolling(window=period, min_periods=period).max().values
    ll = close_s.rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl < 0.0001, 0.0001, range_hl)
    
    # Normalize price to -1 to +1 range
    normalized = 2 * ((close - ll) / range_hl - 0.5)
    normalized = np.clip(normalized, -0.999, 0.999)
    
    # Apply Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    # Signal line (previous fisher value)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

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
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher(close, 9)
    rsi = calculate_rsi(close, 14)
    sma50 = calculate_sma(close, 50)
    hma_4h = calculate_hma(close, 21)
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(sma50[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (HTF)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # 4h trend
        trend_bullish = close[i] > hma_4h[i] and close[i] > sma50[i]
        trend_bearish = close[i] < hma_4h[i] and close[i] < sma50[i]
        
        # Fisher Transform signals (reversal detection)
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_long_cross = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_short_cross = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        
        # Fisher extreme zones
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # RSI confirmation
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: Daily bullish + Fisher oversold cross + RSI confirmation
        if daily_bullish and fisher_long_cross and rsi[i] < 50:
            new_signal = SIZE_ENTRY
        # Path 2: Daily bullish + Fisher extreme oversold + trend bullish
        elif daily_bullish and fisher_oversold and trend_bullish:
            new_signal = SIZE_ENTRY
        # Path 3: Trend bullish + Fisher cross + RSI oversold
        elif trend_bullish and fisher_long_cross and rsi_oversold:
            new_signal = SIZE_ENTRY
        # Path 4: Daily bullish + Price > SMA50 + Fisher turning up
        elif daily_bullish and close[i] > sma50[i] and fisher[i] > fisher_signal[i] and fisher[i] < 0:
            new_signal = SIZE_ENTRY
        # Path 5: Both HMA bullish + Fisher recovering from extreme
        elif close[i] > hma_4h[i] and close[i] > hma_1d_aligned[i] and fisher[i] > -1.0 and fisher_signal[i] < -1.0:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: Daily bearish + Fisher overbought cross + RSI confirmation
        if daily_bearish and fisher_short_cross and rsi[i] > 50:
            new_signal = -SIZE_ENTRY
        # Path 2: Daily bearish + Fisher extreme overbought + trend bearish
        elif daily_bearish and fisher_overbought and trend_bearish:
            new_signal = -SIZE_ENTRY
        # Path 3: Trend bearish + Fisher cross + RSI overbought
        elif trend_bearish and fisher_short_cross and rsi_overbought:
            new_signal = -SIZE_ENTRY
        # Path 4: Daily bearish + Price < SMA50 + Fisher turning down
        elif daily_bearish and close[i] < sma50[i] and fisher[i] < fisher_signal[i] and fisher[i] > 0:
            new_signal = -SIZE_ENTRY
        # Path 5: Both HMA bearish + Fisher falling from extreme
        elif close[i] < hma_4h[i] and close[i] < hma_1d_aligned[i] and fisher[i] < 1.0 and fisher_signal[i] > 1.0:
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