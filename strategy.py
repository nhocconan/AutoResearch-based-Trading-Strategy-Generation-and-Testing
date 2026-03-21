#!/usr/bin/env python3
"""
Experiment #431: 12h Fisher Transform + Daily HMA Trend + Choppiness Regime + ATR Stop
Hypothesis: Fisher Transform excels at catching reversals in bear/range markets (2022 crash, 2025 bear).
Combined with 1d HMA for trend bias and Choppiness Index to distinguish trend vs range regimes,
this should generate >=10 trades/symbol while maintaining positive Sharpe in all market conditions.
Key insight: Fisher Transform normalizes price to Gaussian distribution, making extremes clearer.
In range markets (CHOP>61.8), use Fisher mean-reversion. In trend markets (CHOP<38.2), use Fisher trend-following.
Timeframe: 12h (REQUIRED), HTF: 1d for trend bias via mtf_data helper.
Position size: 0.25 discrete, stoploss 2.5*ATR for 12h timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_daily_hma_chop_regime_atr_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    Excellent for catching reversals in bear markets.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        # Calculate price range
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Avoid division by zero
        if highest == lowest:
            continue
        
        # Normalize price to 0-1 range
        price_norm = 0.66 * ((hl2 - lowest) / (highest - lowest)) + 0.67
        
        # Clamp to avoid log(0) or log(inf)
        price_norm = np.clip(price_norm, 0.001, 0.999)
        
        # Calculate Fisher value
        fisher[i] = 0.5 * np.log((1 + price_norm) / (1 - price_norm))
        
        # Trigger is previous Fisher value
        if i > period:
            trigger[i] = fisher[i-1]
    
    return fisher, trigger

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index - identifies trending vs ranging markets.
    CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend (trend follow).
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        # Highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            atr_sum += tr
        
        # Avoid division by zero
        if highest == lowest or atr_sum == 0:
            continue
        
        # Calculate Choppiness Index
        chop[i] = 100 * np.log10((highest - lowest) / atr_sum) / np.log10(period)
    
    return chop

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
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    sma50 = calculate_sma(close, 50)
    
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
        if np.isnan(atr[i]) or np.isnan(fisher[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma50[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (long-term direction)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # 12h trend filter
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        
        # Choppiness regime
        is_range = chop[i] > 61.8  # Mean reversion regime
        is_trend = chop[i] < 38.2  # Trend following regime
        
        # Fisher Transform signals
        fisher_bullish_cross = fisher[i] > -1.5 and fisher_trigger[i] < -1.5 if not np.isnan(fisher_trigger[i]) else False
        fisher_bearish_cross = fisher[i] < 1.5 and fisher_trigger[i] > 1.5 if not np.isnan(fisher_trigger[i]) else False
        
        # Fisher extreme levels (mean reversion)
        fisher_oversold = fisher[i] < -2.0
        fisher_overbought = fisher[i] > 2.0
        
        # Fisher turning (reversal from extreme)
        fisher_turning_long = fisher[i] > fisher[i-1] and fisher[i-1] < -1.0 if i > 0 and not np.isnan(fisher[i-1]) else False
        fisher_turning_short = fisher[i] < fisher[i-1] and fisher[i-1] > 1.0 if i > 0 and not np.isnan(fisher[i-1]) else False
        
        # RSI confirmation
        rsi_ok_long = rsi[i] > 35 and rsi[i] < 80
        rsi_ok_short = rsi[i] > 20 and rsi[i] < 65
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Fisher bullish cross + Daily bullish + RSI ok (trend regime)
        if is_trend and fisher_bullish_cross and daily_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Path 2: Fisher turning long + Daily bullish + Above SMA50 (range regime)
        elif is_range and fisher_turning_long and daily_bullish and above_sma50:
            new_signal = SIZE_ENTRY
        # Path 3: Fisher oversold + Daily bullish + RSI < 70 (mean reversion)
        elif fisher_oversold and daily_bullish and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        # Path 4: Fisher bullish cross + Above SMA50 + RSI momentum
        elif fisher_bullish_cross and above_sma50 and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        # Path 5: Simple - Daily bullish + Above SMA50 + Fisher > -1.0
        elif daily_bullish and above_sma50 and fisher[i] > -1.0 and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Path 6: Fisher turning + RSI oversold bounce (any regime)
        elif fisher_turning_long and rsi[i] < 45 and rsi[i] > 25:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Fisher bearish cross + Daily bearish + RSI ok (trend regime)
        if is_trend and fisher_bearish_cross and daily_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Path 2: Fisher turning short + Daily bearish + Below SMA50 (range regime)
        elif is_range and fisher_turning_short and daily_bearish and below_sma50:
            new_signal = -SIZE_ENTRY
        # Path 3: Fisher overbought + Daily bearish + RSI > 30 (mean reversion)
        elif fisher_overbought and daily_bearish and rsi[i] > 30:
            new_signal = -SIZE_ENTRY
        # Path 4: Fisher bearish cross + Below SMA50 + RSI momentum
        elif fisher_bearish_cross and below_sma50 and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        # Path 5: Simple - Daily bearish + Below SMA50 + Fisher < 1.0
        elif daily_bearish and below_sma50 and fisher[i] < 1.0 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 6: Fisher turning + RSI overbought drop (any regime)
        elif fisher_turning_short and rsi[i] > 55 and rsi[i] < 75:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest for 12h timeframe)
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
            
            # Calculate trailing stop (2.5*ATR from lowest for 12h timeframe)
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