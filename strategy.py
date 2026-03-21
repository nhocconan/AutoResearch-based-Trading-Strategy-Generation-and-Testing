#!/usr/bin/env python3
"""
Experiment #455: 12h Supertrend + Daily HMA Bias + RSI Pullback + ADX Filter
Hypothesis: 12h timeframe balances trade frequency with noise reduction.
Supertrend provides clear trend direction with ATR-based stops. Daily HMA gives
HTF bias filter. RSI pullback ensures we enter on dips (not chasing tops).
ADX > 20 filters out dead chop without being too restrictive (unlike ADX > 40).
Multiple entry paths ensure >=10 trades requirement is met.
Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper.
Position sizing: 0.30 entry, 0.15 partial exit, 3*ATR stoploss for 12h bars.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_supertrend_daily_hma_rsi_adx_atr_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    n = len(close)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    final_upper[0] = upper_band[0]
    final_lower[0] = lower_band[0]
    supertrend[0] = lower_band[0]
    
    for i in range(1, n):
        # Calculate final upper/lower bands
        if upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = final_upper[i-1]
        
        if lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = final_lower[i-1]
        
        # Determine trend and supertrend value
        if close[i] <= final_upper[i]:
            trend[i] = -1
            supertrend[i] = final_upper[i]
        else:
            trend[i] = 1
            supertrend[i] = final_lower[i]
    
    return supertrend, trend

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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = pd.Series(100 * plus_dm / np.where(atr > 0, atr, 1)).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(100 * minus_dm / np.where(atr > 0, atr, 1)).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

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
    supertrend, st_trend = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    rsi = calculate_rsi(close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    hma_12h = calculate_hma(close, 21)
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (HTF)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # 12h Supertrend direction
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        # 12h HMA trend
        hma_bullish = close[i] > hma_12h[i]
        hma_bearish = close[i] < hma_12h[i]
        
        # ADX filter (trend strength, not too strict)
        trend_present = adx[i] > 20
        
        # RSI pullback zones (entry on dips in uptrend, rallies in downtrend)
        rsi_pullback_long = rsi[i] > 35 and rsi[i] < 55
        rsi_pullback_short = rsi[i] > 45 and rsi[i] < 65
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # DI crossover confirmation
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: Daily bullish + Supertrend bullish + RSI pullback + ADX present
        if daily_bullish and st_bullish and rsi_pullback_long and trend_present:
            new_signal = SIZE_ENTRY
        # Path 2: Daily bullish + Supertrend bullish + DI bullish + RSI > 40
        elif daily_bullish and st_bullish and di_bullish and rsi[i] > 40 and rsi[i] < 60:
            new_signal = SIZE_ENTRY
        # Path 3: Supertrend bullish + HMA bullish + RSI oversold (deep pullback)
        elif st_bullish and hma_bullish and rsi_oversold:
            new_signal = SIZE_ENTRY
        # Path 4: Daily bullish + Supertrend bullish + Price above HMA
        elif daily_bullish and st_bullish and close[i] > hma_12h[i] and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        # Path 5: All trend aligned + RSI neutral (consolidation breakout)
        elif daily_bullish and st_bullish and hma_bullish and di_bullish and rsi[i] > 45 and rsi[i] < 55:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: Daily bearish + Supertrend bearish + RSI pullback + ADX present
        if daily_bearish and st_bearish and rsi_pullback_short and trend_present:
            new_signal = -SIZE_ENTRY
        # Path 2: Daily bearish + Supertrend bearish + DI bearish + RSI < 60
        elif daily_bearish and st_bearish and di_bearish and rsi[i] > 40 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 3: Supertrend bearish + HMA bearish + RSI overbought (rally short)
        elif st_bearish and hma_bearish and rsi_overbought:
            new_signal = -SIZE_ENTRY
        # Path 4: Daily bearish + Supertrend bearish + Price below HMA
        elif daily_bearish and st_bearish and close[i] < hma_12h[i] and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        # Path 5: All trend aligned + RSI neutral (consolidation breakdown)
        elif daily_bearish and st_bearish and hma_bearish and di_bearish and rsi[i] > 45 and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (3*ATR for 12h timeframe - wider stops)
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
            
            # Calculate trailing stop (3*ATR for 12h timeframe - wider stops)
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