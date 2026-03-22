#!/usr/bin/env python3
"""
Experiment #371: 12h KAMA Adaptive Crossover + Daily HMA Trend + Choppiness Regime + ATR Stop
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency - fast during trends,
slow during ranges. This should reduce whipsaws compared to fixed EMA/HMA. 12h timeframe captures
medium-term moves with fewer false signals than 4h. Daily HMA provides trend bias filter.
Choppiness Index (CHOP) detects regime: CHOP>61.8 = range (avoid trend trades), CHOP<38.2 = trend.
RSI(14) with loose thresholds (30-70) ensures minimum trade frequency. ATR(14) stoploss at 2.5x.
Timeframe: 12h (REQUIRED), HTF: 1d for trend bias via mtf_data helper.
Target: Beat Sharpe=0.499 with 40-80 trades total across train+test.
Key insight: KAMA's adaptive nature should outperform fixed MA in crypto's varying volatility regimes.
Build on #359 by replacing Donchian with KAMA crossover + CHOP regime filter.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_daily_hma_chop_regime_rsi_atr_v1"
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

def calculate_kama(close, fast_period=2, slow_period=30, efficiency_period=10):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts speed based on market efficiency (trend vs noise).
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate price change and noise
    price_change = np.abs(close - np.roll(close, efficiency_period))
    price_change[:efficiency_period] = np.abs(close[:efficiency_period] - close[0])
    
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = np.abs(close[i] - close[i-1])
    
    # Sum of noise over efficiency period
    noise_sum = np.zeros(n)
    for i in range(efficiency_period, n):
        noise_sum[i] = np.sum(noise[i-efficiency_period+1:i+1])
    noise_sum[:efficiency_period] = noise_sum[efficiency_period]
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    mask = noise_sum > 0
    er[mask] = price_change[mask] / noise_sum[mask]
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[0] = close[0]
    for i in range(1, n):
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = np.abs(high[j] - close[j-1])
            tr3 = np.abs(low[j] - close[j-1])
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
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    
    # KAMA fast and slow
    kama_fast = calculate_kama(close, fast_period=2, slow_period=10, efficiency_period=10)
    kama_slow = calculate_kama(close, fast_period=2, slow_period=30, efficiency_period=10)
    
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
        
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias
        daily_bullish = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        daily_bearish = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # Choppiness regime filter
        is_trending = chop[i] < 50  # Loose filter to allow more trades
        is_ranging = chop[i] >= 50
        
        # KAMA crossover signals
        kama_cross_long = kama_fast[i] > kama_slow[i] and kama_fast[i-1] <= kama_slow[i-1]
        kama_cross_short = kama_fast[i] < kama_slow[i] and kama_fast[i-1] >= kama_slow[i-1]
        
        # KAMA position (already crossed)
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # RSI filter (LOOSE to ensure trade frequency)
        rsi_ok_long = rsi[i] > 30  # Not deeply oversold
        rsi_ok_short = rsi[i] < 70  # Not deeply overbought
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 40 and rsi[i] < 75
        rsi_momentum_short = rsi[i] > 25 and rsi[i] < 60
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: KAMA cross long + Daily bullish + Trending + RSI ok
        if kama_cross_long and daily_bullish and is_trending and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Secondary: KAMA bullish + Daily bullish + RSI momentum (no cross needed)
        elif kama_bullish and daily_bullish and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Tertiary: KAMA cross long + RSI ok (daily neutral ok in strong trend)
        elif kama_cross_long and is_trending and rsi[i] > 35:
            new_signal = SIZE_ENTRY
        # Quaternary: KAMA bullish alone (ensures minimum trade frequency)
        elif kama_bullish and rsi[i] > 35 and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: KAMA cross short + Daily bearish + Trending + RSI ok
        if kama_cross_short and daily_bearish and is_trending and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Secondary: KAMA bearish + Daily bearish + RSI momentum (no cross needed)
        elif kama_bearish and daily_bearish and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: KAMA cross short + RSI ok (daily neutral ok in strong trend)
        elif kama_cross_short and is_trending and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        # Quaternary: KAMA bearish alone (ensures minimum trade frequency)
        elif kama_bearish and rsi[i] > 30 and rsi[i] < 65:
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