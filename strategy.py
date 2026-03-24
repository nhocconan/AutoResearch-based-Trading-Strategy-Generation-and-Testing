#!/usr/bin/env python3
"""
Experiment #335: 6h Primary + 12h/1d HTF — Ehlers Fisher Transform + HMA Slope Regime

Hypothesis: Previous 6h strategies failed due to overly strict confluence (5+ filters).
This strategy uses Ehlers Fisher Transform (proven reversal indicator) as PRIMARY entry,
with HMA slope for trend direction and Choppiness for regime filtering.

Key innovations:
1. FISHER TRANSFORM (period=9): Catches reversals in bear/range markets better than RSI
   Long: Fisher crosses above -1.5 (oversold reversal)
   Short: Fisher crosses below +1.5 (overbought reversal)
2. HMA SLOPE (not level): Calculate slope over 3 bars - more responsive than crossover
   Slope > 0 = bullish momentum, Slope < 0 = bearish momentum
3. REGIME ADAPTIVE: CHOP > 55 = mean revert (Fisher only), CHOP < 45 = trend (HMA slope + Fisher)
4. LOOSENED confluence: Only 2-3 filters required (not 5+) to ensure 30-60 trades/year
5. HTF BIAS: 12h HMA slope for intermediate trend, 1d close vs HMA for long-term bias

Position sizing: 0.25 base, 0.30 when 1d HTF aligned (discrete levels)
Stoploss: 2.5x ATR(14) from entry

Target: Sharpe>0.40, DD>-40%, trades>=30 train, trades>=3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_hma_slope_chop_12h1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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

def calculate_hma_slope(hma, lookback=3):
    """Calculate HMA slope over lookback bars"""
    n = len(hma)
    slope = np.zeros(n)
    slope[:] = np.nan
    
    for i in range(lookback, n):
        if not np.isnan(hma[i]) and not np.isnan(hma[i-lookback]):
            slope[i] = (hma[i] - hma[i-lookback]) / hma[i-lookback] * 100.0
    
    return slope

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Highlights turning points better than RSI in bear/range markets
    
    Formula:
    1. Calculate typical price: (High + Low) / 2
    2. Normalize: (Price - Lowest) / (Highest - Lowest)
    3. Transform: 0.66 * ((Norm - 0.5) + 0.67 * PrevTransform)
    4. Fisher: 0.5 * ln((1 + Transform) / (1 - Transform))
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    # Typical price
    typical = (high + low) / 2.0
    
    # Normalize and transform
    transform = np.zeros(n)
    transform[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(typical[i-period+1:i+1])
        lowest = np.min(typical[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            normalized = (typical[i] - lowest) / price_range
            normalized = max(0.001, min(0.999, normalized))  # Clamp to avoid ln(0)
            
            # Ehlers transform formula
            if i == period:
                transform[i] = 0.66 * ((normalized - 0.5) + 0.67 * 0.0)
            else:
                transform[i] = 0.66 * ((normalized - 0.5) + 0.67 * transform[i-1])
            
            # Clamp transform to avoid division by zero
            transform[i] = max(-0.999, min(0.999, transform[i]))
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1.0 + transform[i]) / (1.0 - transform[i]))
            
            # Trigger line (1-bar lag of Fisher)
            if i > period:
                trigger[i] = fisher[i-1]
    
    return fisher, trigger

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range bound
    CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    hma_slope_6h = calculate_hma_slope(hma_6h, lookback=3)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate HTF HMA slopes
    hma_slope_12h = calculate_hma_slope(hma_12h_aligned, lookback=2)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Regime memory for hysteresis (avoid flip-flop)
    prev_regime = 0  # 0=unknown, 1=trending, 2=choppy
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(chop[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION with HYSTERESIS ===
        choppy_threshold = 55.0
        trending_threshold = 45.0
        
        if chop[i] > choppy_threshold:
            current_regime = 2  # choppy
        elif chop[i] < trending_threshold:
            current_regime = 1  # trending
        else:
            current_regime = prev_regime  # use memory
        
        prev_regime = current_regime
        
        # === HTF BIAS (12h and 1d) ===
        htf_12h_bull = not np.isnan(hma_slope_12h[i]) and hma_slope_12h[i] > 0
        htf_12h_bear = not np.isnan(hma_slope_12h[i]) and hma_slope_12h[i] < 0
        
        htf_1d_bull = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else False
        htf_1d_bear = close[i] < hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else False
        
        # === 6h HMA SLOPE ===
        hma_slope_bull = not np.isnan(hma_slope_6h[i]) and hma_slope_6h[i] > 0.05
        hma_slope_bear = not np.isnan(hma_slope_6h[i]) and hma_slope_6h[i] < -0.05
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === FISHER TRANSFORM SIGNALS (PRIMARY ENTRY) ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = False
        if i > 0 and not np.isnan(fisher[i]) and not np.isnan(fisher[i-1]):
            if fisher[i-1] <= -1.5 and fisher[i] > -1.5:
                fisher_long = True
        
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = False
        if i > 0 and not np.isnan(fisher[i]) and not np.isnan(fisher[i-1]):
            if fisher[i-1] >= 1.5 and fisher[i] < 1.5:
                fisher_short = True
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY (mean reversion with Fisher only)
        if current_regime == 2:
            # Long: Fisher reversal + above SMA200 (loose filter)
            if fisher_long and above_sma200:
                desired_signal = SIZE_STRONG if htf_1d_bull else SIZE_BASE
            
            # Short: Fisher reversal + below SMA200 (loose filter)
            elif fisher_short and below_sma200:
                desired_signal = -SIZE_STRONG if htf_1d_bear else -SIZE_BASE
        
        # REGIME 2: TRENDING (Fisher + HMA slope confirmation)
        elif current_regime == 1:
            # Long: Fisher reversal + HMA slope bull + 12h HTF bull
            if fisher_long and hma_slope_bull:
                if htf_12h_bull:
                    desired_signal = SIZE_STRONG if htf_1d_bull else SIZE_BASE
                else:
                    desired_signal = SIZE_BASE
            
            # Short: Fisher reversal + HMA slope bear + 12h HTF bear
            elif fisher_short and hma_slope_bear:
                if htf_12h_bear:
                    desired_signal = -SIZE_STRONG if htf_1d_bear else -SIZE_BASE
                else:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
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
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals