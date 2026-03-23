#!/usr/bin/env python3
"""
Experiment #319: 4h Primary + 1d HTF — Fisher Transform + Regime Adaptive

Hypothesis: Recent failures (#307-#318) show complex multi-condition strategies 
either generate 0 trades (over-filtered) or negative Sharpe (whipsaw in 2022 crash).

This strategy uses PROVEN patterns from literature:
1. Ehlers Fisher Transform (period=9): Catches reversals in bear/range markets
   - Long when Fisher crosses above -1.5 (oversold reversal)
   - Short when Fisher crosses below +1.5 (overbought reversal)
2. Choppiness Index(14) regime filter: 
   - CHOP > 55 = range (use Fisher mean reversion)
   - CHOP < 45 = trend (use Fisher with trend bias only)
3. 1d HMA(21) macro bias: Only trade Fisher signals in direction of 1d trend
4. ATR(14) 2.5x trailing stoploss via signal→0

KEY DIFFERENCES from failed experiments:
- SIMPLER entry logic (Fisher crossover is single clean signal)
- NO complex position tracking state (signal-based exits only)
- LOOSE Fisher thresholds (-1.5/+1.5 vs -1.0/+1.0) to trigger MORE trades
- 1d HMA as DIRECTION FILTER only (not exit trigger - that killed trades)
- Position size 0.30 (conservative for 4h volatility)

TARGET: 25-40 trades/year on 4h, Sharpe > 0.6 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_regime_chop_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - lowest) / (highest - lowest) - 0.33
    Catches reversals at extremes. Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(high + low) / 2  # Use typical price
    
    # Highest high and lowest low over period
    highest = high_s.rolling(window=period, min_periods=period).max().values
    lowest = low_s.rolling(window=period, min_periods=period).min().values
    
    # Normalize price to range -1 to +1
    with np.errstate(divide='ignore', invalid='ignore'):
        x = 0.67 * (close_s.values - lowest) / (highest - lowest + 1e-10) - 0.33
        x = np.clip(x, -0.999, 0.999)  # Prevent log domain errors
        
    # Fisher transform
    fisher = 0.5 * np.log((1 + x) / (1 - x + 1e-10))
    fisher = np.nan_to_num(fisher, nan=0.0)
    
    return fisher

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    > 61.8 = choppy/range, < 38.2 = trending
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher = calculate_fisher_transform(high, low, period=9)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30
    
    # Track Fisher previous value for crossover detection
    fisher_prev = np.roll(fisher, 1)
    fisher_prev[0] = fisher[0]
    
    # Track highest/lowest since entry for trailing stop
    highest_since_long = np.zeros(n)
    lowest_since_short = np.zeros(n)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(fisher[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Range market
        is_trending = chop[i] < 45.0  # Trend market
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Update trailing stop levels
        if i > 0:
            highest_since_long[i] = max(highest_since_long[i-1], close[i])
            lowest_since_short[i] = min(lowest_since_short[i-1] if lowest_since_short[i-1] > 0 else close[i], close[i])
        else:
            highest_since_long[i] = close[i]
            lowest_since_short[i] = close[i]
        
        # Check stoploss first (overrides all entry logic)
        stoploss_triggered = False
        
        # Long stoploss: price drops 2.5*ATR from highest since entry
        if signals[i-1] > 0:  # Was long
            stop_price = highest_since_long[i-1] - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
                desired_signal = 0.0
        
        # Short stoploss: price rises 2.5*ATR from lowest since entry
        if signals[i-1] < 0:  # Was short
            stop_price = lowest_since_short[i-1] + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
                desired_signal = 0.0
        
        # If not stopped out, check for new entries
        if not stoploss_triggered:
            if is_choppy:
                # RANGE REGIME: Fisher mean reversion (both directions allowed)
                # Long: Fisher crosses above -1.5 (oversold reversal)
                if fisher_prev[i] < -1.5 and fisher[i] >= -1.5:
                    desired_signal = POSITION_SIZE
                # Short: Fisher crosses below +1.5 (overbought reversal)
                elif fisher_prev[i] > 1.5 and fisher[i] <= 1.5:
                    desired_signal = -POSITION_SIZE
            
            else:  # is_trending or neutral (45-55)
                # TREND REGIME: Fisher with trend bias only
                # Long: Fisher crosses above -1.5 + price above 1d HMA
                if fisher_prev[i] < -1.5 and fisher[i] >= -1.5 and price_above_hma_1d:
                    desired_signal = POSITION_SIZE
                # Short: Fisher crosses below +1.5 + price below 1d HMA
                elif fisher_prev[i] > 1.5 and fisher[i] <= 1.5 and price_below_hma_1d:
                    desired_signal = -POSITION_SIZE
        
        # === HOLD LOGIC ===
        # If already in position and no stoploss/exit trigger, maintain position
        if signals[i-1] > 0 and desired_signal == 0.0 and not stoploss_triggered:
            # Hold long if macro bias still bullish OR in choppy regime
            if price_above_hma_1d or is_choppy:
                desired_signal = POSITION_SIZE
        
        if signals[i-1] < 0 and desired_signal == 0.0 and not stoploss_triggered:
            # Hold short if macro bias still bearish OR in choppy regime
            if price_below_hma_1d or is_choppy:
                desired_signal = -POSITION_SIZE
        
        signals[i] = desired_signal
    
    return signals