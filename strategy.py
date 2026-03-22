#!/usr/bin/env python3
"""
Experiment #487: 15m Fisher Transform + 4h HMA Trend + Choppiness Regime Filter + ATR Stop
Hypothesis: 15m timeframe can work with STRONG HTF filters to reduce noise.
4h HMA provides dominant trend bias (only trade with 4h direction).
Choppiness Index (CHOP) filters out range markets where 15m gets whipsawed.
Fisher Transform (Ehlers) provides cleaner reversal signals than RSI for entries.
Conservative sizing (0.22) and 3*ATR stops account for 15m noise.
Multiple entry paths ensure >=10 trades while quality filters reduce false signals.
Timeframe: 15m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_fisher_4h_hma_chop_regime_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals better than RSI in bear/range markets.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    hl2 = (high + low) / 2.0
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = fisher[i-1] if i > period else 0.0
            continue
        
        # Normalize price to 0-1 range
        x = (hl2[i] - lowest) / (highest - lowest)
        
        # Clamp to avoid division issues
        x = np.clip(x, 0.001, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        # Signal line (1-period lag)
        if i > period:
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 50.0
            continue
        
        atr_sum = np.sum(atr[i-period+1:i+1])
        
        if atr_sum == 0:
            chop[i] = 50.0
            continue
        
        chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA) - adapts to market noise.
    ER (Efficiency Ratio) determines smoothing constant.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i-er_period])
        noise = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

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
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama = calculate_kama(close, 10, 2, 30)
    
    # 15m HMA for additional confirmation
    hma_15m = calculate_hma(close, 21)
    hma_15m_fast = calculate_hma(close, 10)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.22
    SIZE_HALF = 0.11
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - STRONG FILTER
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 4h HMA slope confirmation
        hma_4h_rising = hma_4h_aligned[i] > hma_4h_aligned[i-1] if i > 0 else False
        hma_4h_falling = hma_4h_aligned[i] < hma_4h_aligned[i-1] if i > 0 else False
        
        # Choppiness regime filter
        chop_range = chop[i] > 55  # Range market
        chop_trend = chop[i] < 45  # Trending market
        
        # Fisher Transform signals
        fisher_long_cross = fisher[i] > -1.5 and fisher_signal[i] <= -1.5 if i > 0 else False
        fisher_short_cross = fisher[i] < 1.5 and fisher_signal[i] >= 1.5 if i > 0 else False
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        fisher_turning_up = fisher[i] > fisher[i-1] if i > 0 else False
        fisher_turning_down = fisher[i] < fisher[i-1] if i > 0 else False
        
        # 15m HMA trend
        hma_15m_bullish = close[i] > hma_15m[i]
        hma_15m_bearish = close[i] < hma_15m[i]
        hma_15m_rising = hma_15m[i] > hma_15m[i-1] if i > 0 else False
        hma_15m_falling = hma_15m[i] < hma_15m[i-1] if i > 0 else False
        
        # Fast HMA crossover
        fast_above_slow = hma_15m_fast[i] > hma_15m[i]
        fast_below_slow = hma_15m_fast[i] < hma_15m[i]
        
        # KAMA trend
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # RSI zones
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: 4h bullish + trending regime + Fisher long cross + 15m HMA bullish
        if hma_4h_bullish and chop_trend and fisher_long_cross and hma_15m_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 2: 4h bullish + 4h rising + Fisher oversold turning up + RSI neutral
        elif hma_4h_bullish and hma_4h_rising and fisher_oversold and fisher_turning_up and rsi_neutral:
            new_signal = SIZE_ENTRY
        
        # Path 3: 4h bullish + 15m HMA rising + Fast HMA crossover up + KAMA bullish
        elif hma_4h_bullish and hma_15m_rising and fast_above_slow and kama_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 4: 4h bullish + Fisher oversold bounce + RSI oversold (deep pullback entry)
        elif hma_4h_bullish and fisher_oversold and rsi_oversold and fisher_turning_up:
            new_signal = SIZE_ENTRY
        
        # Path 5: 4h bullish + 15m HMA bullish + chop neutral + Fisher turning up
        elif hma_4h_bullish and hma_15m_bullish and chop[i] < 55 and fisher_turning_up:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: 4h bearish + trending regime + Fisher short cross + 15m HMA bearish
        if hma_4h_bearish and chop_trend and fisher_short_cross and hma_15m_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 2: 4h bearish + 4h falling + Fisher overbought turning down + RSI neutral
        elif hma_4h_bearish and hma_4h_falling and fisher_overbought and fisher_turning_down and rsi_neutral:
            new_signal = -SIZE_ENTRY
        
        # Path 3: 4h bearish + 15m HMA falling + Fast HMA crossover down + KAMA bearish
        elif hma_4h_bearish and hma_15m_falling and fast_below_slow and kama_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 4: 4h bearish + Fisher overbought drop + RSI overbought (deep rally entry)
        elif hma_4h_bearish and fisher_overbought and rsi_overbought and fisher_turning_down:
            new_signal = -SIZE_ENTRY
        
        # Path 5: 4h bearish + 15m HMA bearish + chop neutral + Fisher turning down
        elif hma_4h_bearish and hma_15m_bearish and chop[i] < 55 and fisher_turning_down:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (3*ATR for 15m timeframe - wider for noise)
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
            
            # Calculate trailing stop (3*ATR for 15m timeframe)
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