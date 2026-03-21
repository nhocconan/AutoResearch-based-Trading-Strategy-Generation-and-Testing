#!/usr/bin/env python3
"""
Experiment #388: 4h Fisher Transform + Choppiness Regime + Daily HMA Trend + ATR Stop
Hypothesis: Fisher Transform excels at catching reversals in bear/range markets (2025 test period).
Combined with Choppiness Index regime detection, we can switch between mean-reversion (range)
and trend-following (trending) logic. Daily HMA provides overall trend bias. This is DIFFERENT
from failed supertrend/RSI combinations (#382, #376, #379). Fisher crosses extreme levels (-1.5/+1.5)
signal reversals. Choppiness > 61.8 = range (use Fisher mean-reversion), < 38.2 = trend (use Fisher
with trend bias). Position size 0.25 discrete, stoploss 2.5*ATR. Target: Beat Sharpe=0.499.
Timeframe: 4h (REQUIRED), HTF: 1d via mtf_data helper (call ONCE before loop).
Key insight: Fisher Transform + regime filter = adaptive strategy for bear/range 2025 market.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_regime_daily_hma_atr_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution, extreme values signal reversals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    # Calculate median price
    median = (high + low) / 2
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        range_val = highest - lowest
        if range_val == 0:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            trigger[i] = fisher[i]
            continue
        
        # Normalize price to -1 to +1 range
        normalized = 2 * (median[i] - lowest) / range_val - 1
        
        # Apply exponential smoothing
        if i == period:
            smooth = normalized
        else:
            smooth = 0.67 * normalized + 0.33 * (2 * (median[i-1] - lowest) / range_val - 1)
        
        # Clamp to avoid division issues
        smooth = np.clip(smooth, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + smooth) / (1 - smooth))
        if i > period:
            fisher[i] = 0.67 * fisher[i] + 0.33 * fisher[i-1]
        
        trigger[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, trigger

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (use mean-reversion)
    CHOP < 38.2 = trending market (use trend-following)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = np.abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = np.abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            tr_sum += tr
        
        if highest_high - lowest_low > 0:
            chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop[:period] = 50.0
    return chop

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
    chop = calculate_choppiness(high, low, close, 14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, 9)
    rsi = calculate_rsi(close, 14)
    
    # Additional trend indicator for trending regime
    hma_fast = calculate_hma(close, 8)
    hma_slow = calculate_hma(close, 21)
    
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
        if np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias
        daily_bullish = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        daily_bearish = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # Regime detection
        is_ranging = chop[i] > 55  # More lenient than 61.8 to ensure trades
        is_trending = chop[i] <= 55
        
        # Fisher Transform signals
        fisher_cross_long = fisher[i] > -1.5 and fisher_trigger[i] <= -1.5
        fisher_cross_short = fisher[i] < 1.5 and fisher_trigger[i] >= 1.5
        fisher_extreme_long = fisher[i] < -1.8  # Deep oversold
        fisher_extreme_short = fisher[i] > 1.8  # Deep overbought
        
        # HMA trend
        hma_bullish = hma_fast[i] > hma_slow[i]
        hma_bearish = hma_fast[i] < hma_slow[i]
        hma_cross_long = hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]
        hma_cross_short = hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]
        
        new_signal = 0.0
        
        # === RANGING REGIME (Mean Reversion with Fisher) ===
        if is_ranging:
            # Long: Fisher extreme oversold + RSI not too high
            if fisher_extreme_long and rsi[i] < 60:
                new_signal = SIZE_ENTRY
            # Long: Fisher cross up from extreme
            elif fisher_cross_long and rsi[i] < 65:
                new_signal = SIZE_ENTRY
            # Long: Fisher turning up from below -1.0
            elif fisher[i] > fisher_trigger[i] and fisher[i] < -0.5 and rsi[i] < 55:
                new_signal = SIZE_ENTRY
            
            # Short: Fisher extreme overbought + RSI not too low
            if fisher_extreme_short and rsi[i] > 40:
                new_signal = -SIZE_ENTRY
            # Short: Fisher cross down from extreme
            elif fisher_cross_short and rsi[i] > 35:
                new_signal = -SIZE_ENTRY
            # Short: Fisher turning down from above +1.0
            elif fisher[i] < fisher_trigger[i] and fisher[i] > 0.5 and rsi[i] > 45:
                new_signal = -SIZE_ENTRY
        
        # === TRENDING REGIME (Trend Following with Fisher + Daily Bias) ===
        elif is_trending:
            # Long: Fisher cross long + Daily bullish + HMA bullish
            if fisher_cross_long and daily_bullish and hma_bullish:
                new_signal = SIZE_ENTRY
            # Long: Fisher not extreme short + Daily bullish + HMA cross long
            elif fisher[i] > -1.5 and daily_bullish and hma_cross_long:
                new_signal = SIZE_ENTRY
            # Long: Fisher turning up + Daily bullish (weaker signal)
            elif fisher[i] > fisher_trigger[i] and daily_bullish and rsi[i] > 40:
                new_signal = SIZE_ENTRY
            # Long: HMA cross long alone (ensures trade frequency)
            elif hma_cross_long and rsi[i] > 35 and rsi[i] < 70:
                new_signal = SIZE_ENTRY
            
            # Short: Fisher cross short + Daily bearish + HMA bearish
            if fisher_cross_short and daily_bearish and hma_bearish:
                new_signal = -SIZE_ENTRY
            # Short: Fisher not extreme long + Daily bearish + HMA cross short
            elif fisher[i] < 1.5 and daily_bearish and hma_cross_short:
                new_signal = -SIZE_ENTRY
            # Short: Fisher turning down + Daily bearish (weaker signal)
            elif fisher[i] < fisher_trigger[i] and daily_bearish and rsi[i] < 60:
                new_signal = -SIZE_ENTRY
            # Short: HMA cross short alone (ensures trade frequency)
            elif hma_cross_short and rsi[i] > 30 and rsi[i] < 65:
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