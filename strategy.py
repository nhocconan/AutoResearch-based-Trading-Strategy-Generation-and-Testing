#!/usr/bin/env python3
"""
Experiment #370: 4h KAMA Adaptive Trend + Daily/Weekly HMA + Choppiness Regime + RSI Momentum
Hypothesis: 4h timeframe needs adaptive indicators (KAMA) that adjust to market efficiency.
Choppiness Index detects regime (trending vs range) to switch logic. Daily HMA provides
intermediate trend bias, Weekly HMA provides macro filter. RSI(14) with loose thresholds
(30-70) ensures trade frequency. ATR(14) stoploss at 2.0x protects capital.
Key insight from failures: 4h strategies fail due to whipsaws - KAMA adapts to volatility,
Choppiness filters out range markets where trend strategies fail. Multiple entry paths
ensure minimum trade frequency (critical - many 4h strategies got 0 trades).
Timeframe: 4h (REQUIRED), HTF: 1d + 1w for trend bias via mtf_data helper.
Target: Beat Sharpe=0.499 with 40-80 trades total across train+test.
Position sizing: 0.25 entry, 0.125 half-position at take profit.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adaptive_daily_weekly_hma_chop_regime_rsi_atr_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.abs(close[:er_period] - close[0])
    
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    volatility[:er_period] = change[:er_period]
    
    er = np.zeros(n)
    er[er_period:] = np.where(volatility[er_period:] > 0, change[er_period:] / volatility[er_period:], 0)
    er[:er_period] = 1.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc = np.clip(sc, slow_sc, fast_sc)
    
    # Initialize KAMA
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
    """Calculate Choppiness Index (CHOP) for regime detection."""
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        if highest_high - lowest_low > 0:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    chop = calculate_choppiness(high, low, close, 14)
    
    # KAMA fast line for crossover signals
    kama_fast = calculate_kama(close, er_period=5, fast_period=2, slow_period=15)
    
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
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(kama[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # Weekly macro trend bias (SOFT filter)
        weekly_bullish = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        weekly_bearish = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # Daily intermediate trend bias
        daily_bullish = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        daily_bearish = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # Regime detection via Choppiness Index
        # CHOP > 61.8 = range/choppy (mean reversion mode)
        # CHOP < 38.2 = trending (trend following mode)
        is_trending = chop[i] < 45.0
        is_choppy = chop[i] > 55.0
        
        # KAMA trend signals
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # KAMA crossover signals
        kama_cross_long = kama_fast[i] > kama[i] and kama_fast[i-1] <= kama[i-1]
        kama_cross_short = kama_fast[i] < kama[i] and kama_fast[i-1] >= kama[i-1]
        
        # RSI momentum (LOOSE thresholds for trade frequency)
        rsi_ok_long = rsi[i] > 30 and rsi[i] < 70
        rsi_ok_short = rsi[i] > 30 and rsi[i] < 70
        rsi_strong_long = rsi[i] > 40 and rsi[i] < 65
        rsi_strong_short = rsi[i] > 35 and rsi[i] < 60
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure trade frequency) ===
        
        # Path 1: Trending regime + KAMA bullish + Daily bullish + RSI ok
        if is_trending and kama_bullish and daily_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        
        # Path 2: KAMA crossover long + Daily bullish (regime neutral)
        elif kama_cross_long and daily_bullish and rsi[i] > 35:
            new_signal = SIZE_ENTRY
        
        # Path 3: Choppy regime + KAMA bullish + RSI not overbought (mean reversion long)
        elif is_choppy and kama_bullish and rsi[i] > 30 and rsi[i] < 55:
            new_signal = SIZE_ENTRY
        
        # Path 4: Weekly bullish + KAMA bullish + RSI ok (macro bias)
        elif weekly_bullish and kama_bullish and rsi[i] > 35 and rsi[i] < 65:
            new_signal = SIZE_ENTRY
        
        # Path 5: KAMA cross long alone (ensures minimum trades)
        elif kama_cross_long and rsi[i] > 30 and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure trade frequency) ===
        
        # Path 1: Trending regime + KAMA bearish + Daily bearish + RSI ok
        if is_trending and kama_bearish and daily_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        
        # Path 2: KAMA crossover short + Daily bearish (regime neutral)
        elif kama_cross_short and daily_bearish and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        
        # Path 3: Choppy regime + KAMA bearish + RSI not oversold (mean reversion short)
        elif is_choppy and kama_bearish and rsi[i] > 45 and rsi[i] < 70:
            new_signal = -SIZE_ENTRY
        
        # Path 4: Weekly bearish + KAMA bearish + RSI ok (macro bias)
        elif weekly_bearish and kama_bearish and rsi[i] > 35 and rsi[i] < 65:
            new_signal = -SIZE_ENTRY
        
        # Path 5: KAMA cross short alone (ensures minimum trades)
        elif kama_cross_short and rsi[i] > 30 and rsi[i] < 70:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest)
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
            
            # Calculate trailing stop (2.0*ATR from lowest)
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
        prev_signal = signals[i-1] if i > 0 else 0.0
        
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
    
    return signals