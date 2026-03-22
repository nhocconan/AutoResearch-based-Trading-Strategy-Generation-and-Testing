#!/usr/bin/env python3
"""
Experiment #468: Daily HMA Trend + Weekly Bias + RSI Mean Reversion + ATR Stop

Hypothesis: Daily timeframe captures major crypto cycles while weekly HTF filter 
prevents counter-trend trades. RSI mean reversion entries (oversold in uptrend, 
overbought in downtrend) work well in crypto's volatile nature. Multiple entry 
paths ensure >=10 trades per symbol. 3*ATR stoploss accommodates daily volatility.

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (get_htf_data called ONCE before loop)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_weekly_bias_rsi_mr_atr_v1"
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

def calculate_slope(values, lookback=5):
    """Calculate slope of values over lookback period."""
    n = len(values)
    slope = np.zeros(n)
    slope[:] = np.nan
    for i in range(lookback, n):
        if not np.isnan(values[i]) and not np.isnan(values[i - lookback]):
            slope[i] = (values[i] - values[i - lookback]) / lookback
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    hma_1d = calculate_hma(close, 21)
    hma_1d_fast = calculate_hma(close, 9)
    rsi = calculate_rsi(close, 14)
    hma_slope = calculate_slope(hma_1d, lookback=5)
    
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
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(hma_slope[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias (HTF)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Daily HMA trend
        daily_bullish = close[i] > hma_1d[i]
        daily_bearish = close[i] < hma_1d[i]
        hma_rising = hma_slope[i] > 0
        hma_falling = hma_slope[i] < 0
        
        # Fast HMA crossover
        fast_above_slow = hma_1d_fast[i] > hma_1d[i]
        fast_below_slow = hma_1d_fast[i] < hma_1d[i]
        
        # RSI mean reversion zones (LOOSENED for more trades)
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_neutral_long = rsi[i] > 35 and rsi[i] < 55
        rsi_neutral_short = rsi[i] > 45 and rsi[i] < 65
        rsi_extreme_long = rsi[i] < 30
        rsi_extreme_short = rsi[i] > 70
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: Weekly bullish + Daily bullish + RSI oversold (pullback entry)
        if weekly_bullish and daily_bullish and rsi_oversold:
            new_signal = SIZE_ENTRY
        # Path 2: Weekly bullish + Fast HMA above slow + RSI neutral
        elif weekly_bullish and fast_above_slow and rsi_neutral_long:
            new_signal = SIZE_ENTRY
        # Path 3: Daily bullish + HMA rising + RSI > 35 (momentum continuation)
        elif daily_bullish and hma_rising and rsi[i] > 35 and rsi[i] < 55:
            new_signal = SIZE_ENTRY
        # Path 4: Fast HMA crossover up + RSI > 40 (momentum entry)
        elif fast_above_slow and hma_1d_fast[i] > hma_1d_fast[i-1] and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Path 5: RSI very oversold (<30) - deep mean reversion regardless of trend
        elif rsi_extreme_long:
            new_signal = SIZE_ENTRY
        # Path 6: Weekly bullish + Daily bullish + Price above both HMA (trend follow)
        elif weekly_bullish and daily_bullish and close[i] > hma_1d[i]:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: Weekly bearish + Daily bearish + RSI overbought (rally short)
        if weekly_bearish and daily_bearish and rsi_overbought:
            new_signal = -SIZE_ENTRY
        # Path 2: Weekly bearish + Fast HMA below slow + RSI neutral
        elif weekly_bearish and fast_below_slow and rsi_neutral_short:
            new_signal = -SIZE_ENTRY
        # Path 3: Daily bearish + HMA falling + RSI < 65 (momentum continuation)
        elif daily_bearish and hma_falling and rsi[i] > 45 and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        # Path 4: Fast HMA crossover down + RSI < 60 (momentum entry)
        elif fast_below_slow and hma_1d_fast[i] < hma_1d_fast[i-1] and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 5: RSI very overbought (>70) - deep mean reversion regardless of trend
        elif rsi_extreme_short:
            new_signal = -SIZE_ENTRY
        # Path 6: Weekly bearish + Daily bearish + Price below both HMA (trend follow)
        elif weekly_bearish and daily_bearish and close[i] < hma_1d[i]:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (3*ATR for daily timeframe)
            current_stop = highest_close - 3.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (3*ATR for daily timeframe)
            current_stop = lowest_close + 3.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
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
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
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