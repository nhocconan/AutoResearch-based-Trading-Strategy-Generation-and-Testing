#!/usr/bin/env python3
"""
Experiment #383: 12h HMA Crossover + Daily HMA Trend + Choppiness Regime + RSI + ATR Stop
Hypothesis: HMA (Hull Moving Average) has less lag than EMA while maintaining smoothness.
On 12h timeframe, HMA crossover should capture medium-term trends with fewer whipsaws than
the KAMA approach (#371 Sharpe=0.100). Daily HMA provides trend bias. Choppiness Index
filters out ranging markets where crossovers fail. RSI(14) with loose thresholds (35-65)
ensures minimum trade frequency (critical - many strategies failed with 0 trades).
ATR(14) stoploss at 2.5x protects capital. Position size 0.30 discrete to minimize fees.
Timeframe: 12h (REQUIRED), HTF: 1d for trend bias via mtf_data helper (call ONCE before loop).
Target: Beat Sharpe=0.499 (current best mtf_12h_supertrend_daily_hma_rsi_pullback_v2).
Key insight: Simpler HMA crossover + looser filters = more trades + better Sharpe than complex KAMA.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_daily_hma_chop_regime_rsi_atr_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (avoid trend trades)
    CHOP < 38.2 = trending market (enter on signals)
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
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    
    # HMA fast and slow for crossover
    hma_fast = calculate_hma(close, 8)
    hma_slow = calculate_hma(close, 21)
    
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
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias
        daily_bullish = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        daily_bearish = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # Choppiness regime filter (looser to ensure trades)
        is_trending = chop[i] < 55  # More lenient than 50
        is_ranging = chop[i] >= 55
        
        # HMA crossover signals
        hma_cross_long = hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]
        hma_cross_short = hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]
        
        # HMA position (already crossed)
        hma_bullish = hma_fast[i] > hma_slow[i]
        hma_bearish = hma_fast[i] < hma_slow[i]
        
        # RSI filter (LOOSE to ensure trade frequency - CRITICAL)
        rsi_ok_long = rsi[i] > 35  # Not deeply oversold
        rsi_ok_short = rsi[i] < 65  # Not deeply overbought
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 40 and rsi[i] < 75
        rsi_momentum_short = rsi[i] > 25 and rsi[i] < 60
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple conditions to ensure trades) ===
        # Primary: HMA cross long + Daily bullish + Trending + RSI ok
        if hma_cross_long and daily_bullish and is_trending and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Secondary: HMA bullish + Daily bullish + RSI momentum (no cross needed)
        elif hma_bullish and daily_bullish and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Tertiary: HMA cross long + RSI ok (daily neutral ok in strong trend)
        elif hma_cross_long and is_trending and rsi[i] > 35:
            new_signal = SIZE_ENTRY
        # Quaternary: HMA bullish alone (ensures minimum trade frequency)
        elif hma_bullish and rsi[i] > 35 and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        # Quintenary: HMA cross long even in ranging (backup)
        elif hma_cross_long and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple conditions to ensure trades) ===
        # Primary: HMA cross short + Daily bearish + Trending + RSI ok
        if hma_cross_short and daily_bearish and is_trending and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Secondary: HMA bearish + Daily bearish + RSI momentum (no cross needed)
        elif hma_bearish and daily_bearish and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: HMA cross short + RSI ok (daily neutral ok in strong trend)
        elif hma_cross_short and is_trending and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        # Quaternary: HMA bearish alone (ensures minimum trade frequency)
        elif hma_bearish and rsi[i] > 30 and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        # Quintenary: HMA cross short even in ranging (backup)
        elif hma_cross_short and rsi[i] < 60:
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