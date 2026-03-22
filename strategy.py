#!/usr/bin/env python3
"""
Experiment #472: 4h Fisher Transform + Daily HMA Bias + Choppiness Regime + ATR Stop
Hypothesis: Fisher Transform catches reversals better than RSI in bear/range markets (2025 test).
Combined with Daily HMA bias filter and Choppiness Index regime detection, this should:
1. Enter on extreme reversals (Fisher <-1.5 long, >+1.5 short)
2. Only trade when regime matches strategy (CHOP>50 = range = mean revert)
3. Align with daily trend bias (long when price>daily_HMA, short when below)
4. Use tight ATR stops (2.0*ATR) to protect capital in volatile 4h bars
Timeframe: 4h (REQUIRED), HTF: 1d via mtf_data helper.
Position sizing: 0.25 discrete levels to minimize fee churn while maintaining exposure.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_daily_hma_chop_regime_atr_v1"
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

def calculate_fisher_transform(close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals at extremes better than RSI in range markets.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    close_s = pd.Series(close)
    # Calculate (high - low) / 2 + (close - prev_close) / 2
    high = close_s  # Use close as proxy when high/low not available in function
    low = close_s
    
    # Price position within recent range
    price_range = close_s.rolling(window=period, min_periods=period).max().values - \
                  close_s.rolling(window=period, min_periods=period).min().values
    price_range = np.where(price_range > 0, price_range, 1e-10)
    
    normalized = (close - close_s.rolling(window=period, min_periods=period).min().values) / price_range
    normalized = np.clip(normalized, 0.001, 0.999)
    
    # Fisher transform
    fisher_input = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
    fisher = pd.Series(fisher_input).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return fisher

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - identifies ranging vs trending markets.
    CHOP > 61.8 = range (mean reversion works)
    CHOP < 38.2 = trend (trend following works)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0:
            tr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], 
                        abs(high[j] - close[j-1]), 
                        abs(low[j] - close[j-1]))
                tr_sum += tr
            
            chop[i] = 100 * np.log10(tr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

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
    fisher = calculate_fisher_transform(close, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
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
    
    # Track Fisher crossings
    prev_fisher = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (HTF)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # Choppiness regime
        is_ranging = chop[i] > 50.0  # Range market - mean reversion works
        is_trending = chop[i] < 45.0  # Trend market - trend following works
        
        # Fisher Transform signals
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher[i] > -1.5 and prev_fisher <= -1.5
        fisher_cross_down = fisher[i] < 1.5 and prev_fisher >= 1.5
        
        # RSI confirmation
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: Daily bullish + Fisher oversold + RSI confirmation (mean reversion in uptrend)
        if daily_bullish and fisher_oversold and rsi_oversold:
            new_signal = SIZE_ENTRY
        # Path 2: Daily bullish + Fisher cross up + RSI < 50 (reversal entry)
        elif daily_bullish and fisher_cross_up and rsi[i] < 50:
            new_signal = SIZE_ENTRY
        # Path 3: Ranging market + Fisher oversold + RSI < 40 (pure mean reversion)
        elif is_ranging and fisher_oversold and rsi[i] < 40:
            new_signal = SIZE_ENTRY
        # Path 4: Daily bullish + RSI oversold + price near daily HMA (pullback)
        elif daily_bullish and rsi_oversold and close[i] < hma_1d_aligned[i] * 1.02:
            new_signal = SIZE_ENTRY
        # Path 5: Fisher cross up + RSI rising (momentum confirmation)
        elif fisher_cross_up and rsi[i] > rsi[i-1] and rsi[i] < 55:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: Daily bearish + Fisher overbought + RSI confirmation
        if daily_bearish and fisher_overbought and rsi_overbought:
            new_signal = -SIZE_ENTRY
        # Path 2: Daily bearish + Fisher cross down + RSI > 50
        elif daily_bearish and fisher_cross_down and rsi[i] > 50:
            new_signal = -SIZE_ENTRY
        # Path 3: Ranging market + Fisher overbought + RSI > 60
        elif is_ranging and fisher_overbought and rsi[i] > 60:
            new_signal = -SIZE_ENTRY
        # Path 4: Daily bearish + RSI overbought + price near daily HMA (rally short)
        elif daily_bearish and rsi_overbought and close[i] > hma_1d_aligned[i] * 0.98:
            new_signal = -SIZE_ENTRY
        # Path 5: Fisher cross down + RSI falling
        elif fisher_cross_down and rsi[i] < rsi[i-1] and rsi[i] > 45:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR for 4h timeframe - tighter than 12h)
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
            
            # Calculate trailing stop (2.0*ATR for 4h timeframe)
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
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
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
        prev_fisher = fisher[i]
    
    return signals