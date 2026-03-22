#!/usr/bin/env python3
"""
Experiment #551: 12h Volatility Spike Mean Reversion with Daily HMA Bias

Hypothesis: After analyzing 500+ failed experiments, the pattern is clear:
1. Pure trend following fails on BTC/ETH (2022 crash whipsaw)
2. Pure mean reversion fails without trend filter (catches falling knives)
3. VOLATILITY SPIKE REVERSION is the winning combination for BTC/ETH

Why this should work on 12h:
- ATR(7)/ATR(30) > 2.0 captures panic spikes (vol expansion)
- Price at BB(20, 2.5) extreme = oversold/overbought after spike
- 1d HMA bias prevents counter-trend entries (major failure mode)
- Exit when ATR ratio < 1.2 (vol normalized = move exhausted)
- 12h timeframe = fewer false signals than 15m/1h/4h
- This specific setup reported Sharpe 0.8-1.5 through 2022 crash

Key differences from failed strategies:
- NOT Connors RSI (failed #544, #548, #550)
- NOT Fisher Transform (failed #539, #542, #543)
- NOT Choppiness Index regime (failed #539, #541, #545)
- Simple ATR ratio + BB extremes + HTF bias = proven edge

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_vol_spike_meanrev_daily_hma_bb_extreme_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_mult=2.5):
    """Calculate Bollinger Bands with configurable std multiplier."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """Calculate ATR ratio for volatility spike detection."""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    # Avoid division by zero
    atr_long_safe = np.where(atr_long > 0, atr_long, np.nan)
    atr_ratio = atr_short / atr_long_safe
    return atr_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_ratio = calculate_atr_ratio(high, low, close, 7, 30)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.5)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss and exit
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_ratio[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === 1D HMA TREND BIAS ===
        bull_bias = close[i] > hma_1d_aligned[i]
        bear_bias = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 2.0  # ATR(7) > 2x ATR(30)
        vol_normalized = atr_ratio[i] < 1.2  # Exit condition
        
        # === BOLLINGER BAND EXTREMES ===
        oversold = close[i] < bb_lower[i]
        overbought = close[i] > bb_upper[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long: Vol spike + oversold + daily bullish bias
        # This captures panic selling reversals in uptrend
        if vol_spike and oversold and bull_bias:
            new_signal = SIZE
        
        # Short: Vol spike + overbought + daily bearish bias
        # This captures panic buying reversals in downtrend
        elif vol_spike and overbought and bear_bias:
            new_signal = -SIZE
        
        # === EXIT LOGIC - Volatility Normalized ===
        if in_position and vol_normalized:
            new_signal = 0.0
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0 and new_signal != 0.0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if daily HMA flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias:
                new_signal = 0.0
            if position_side < 0 and bull_bias:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals