#!/usr/bin/env python3
"""
Experiment #723: 6h Primary + 1d/1w HTF — Simplified Trend + Mean Reversion Hybrid

Hypothesis: 6h timeframe needs LOOSE entry conditions to generate sufficient trades.
Previous 6h strategies failed due to over-filtering (Sharpe=0.000 = 0 trades).

Key innovations:
1. Dual-mode strategy: Trend follow when CHOP<45, mean revert when CHOP>55
2. 1d HMA(21) for intermediate trend bias (less strict than 1w)
3. 1w HMA(21) for major trend confirmation (only filter, not entry trigger)
4. RSI(14) with LOOSE thresholds (35/65 not 30/70) for more trades
5. Simple HMA(21) crossover on 6h for trend entries
6. ATR(14) 2.5x trailing stop
7. Discrete sizing: 0.0, ±0.25, ±0.30

Entry conditions (LOOSE to ensure 30+ trades):
- LONG trend: 1d HMA bull + 6h HMA cross up + CHOP<50
- LONG mean revert: 1d HMA bull + RSI<40 + CHOP>50
- SHORT trend: 1d HMA bear + 6h HMA cross down + CHOP<50
- SHORT mean revert: 1d HMA bear + RSI>60 + CHOP>50

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_hma_rsi_chop_hybrid_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - identifies trending vs ranging markets"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[j-1]) if (j:=i) else abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    hma_21 = calculate_hma(close, period=21)
    hma_48 = calculate_hma(close, period=48)
    rsi = calculate_rsi(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(hma_48[i]) or np.isnan(chop[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d and 1w HMA) ===
        # 1d HMA for intermediate trend (primary filter)
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # 1w HMA for major trend (confirmation only, less strict)
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP < 38.2 = strong trend, 38.2-61.8 = neutral, > 61.8 = range
        # Use loose thresholds for more trades
        trend_regime = chop[i] < 50.0  # Loose: more trend signals
        range_regime = chop[i] > 45.0  # Loose: overlap for more mean revert
        
        # === HMA CROSSOVER (6h) ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_21[i-1]) and not np.isnan(hma_48[i-1]):
            hma_cross_long = (hma_21[i] > hma_48[i]) and (hma_21[i-1] <= hma_48[i-1])
            hma_cross_short = (hma_21[i] < hma_48[i]) and (hma_21[i-1] >= hma_48[i-1])
        
        # === HMA POSITION (current state) ===
        hma_bull = hma_21[i] > hma_48[i]
        hma_bear = hma_21[i] < hma_48[i]
        
        # === RSI EXTREMES (LOOSE for more trades) ===
        rsi_oversold = rsi[i] < 40.0  # Was 30, now 40 for more trades
        rsi_overbought = rsi[i] > 60.0  # Was 70, now 60 for more trades
        rsi_extreme_oversold = rsi[i] < 30.0
        rsi_extreme_overbought = rsi[i] > 70.0
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: Trend regime + 1d bull + HMA crossover or HMA bull
        if trend_regime and htf_1d_bull:
            if hma_cross_long or (hma_bull and rsi[i] < 50):
                desired_signal = SIZE_STRONG
        # LONG: Range regime + 1d bull + RSI oversold
        elif range_regime and htf_1d_bull and rsi_oversold:
            desired_signal = SIZE_BASE
        # LONG: 1d bull + RSI extreme oversold (any regime)
        elif htf_1d_bull and rsi_extreme_oversold:
            desired_signal = SIZE_BASE
        # LONG: 1w bull + HMA bull (major trend follow)
        elif htf_1w_bull and hma_bull and not in_position:
            desired_signal = SIZE_BASE
        
        # SHORT: Trend regime + 1d bear + HMA crossover or HMA bear
        elif trend_regime and htf_1d_bear:
            if hma_cross_short or (hma_bear and rsi[i] > 50):
                desired_signal = -SIZE_STRONG
        # SHORT: Range regime + 1d bear + RSI overbought
        elif range_regime and htf_1d_bear and rsi_overbought:
            desired_signal = -SIZE_BASE
        # SHORT: 1d bear + RSI extreme overbought (any regime)
        elif htf_1d_bear and rsi_extreme_overbought:
            desired_signal = -SIZE_BASE
        # SHORT: 1w bear + HMA bear (major trend follow)
        elif htf_1w_bear and hma_bear and not in_position:
            desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals