#!/usr/bin/env python3
"""
Experiment #568: 4h Fisher Transform Reversal with Dual HTF (1d/1w) Trend Bias

Hypothesis: After 500+ failed experiments, the key insight is:
1. Fisher Transform excels at catching reversals in bear/range markets (2022, 2025)
2. Dual HTF (1d + 1w HMA) provides stronger trend bias than single HTF
3. 4h timeframe balances noise reduction with sufficient trade frequency
4. Volatility regime filter (ATR ratio) avoids entries during extreme vol spikes
5. Asymmetric sizing: smaller positions in uncertain regimes

Why this should work on 4h:
- Fisher Transform normalizes price to Gaussian distribution, better than RSI for reversals
- 1d + 1w HMA alignment ensures we only trade with major trend
- ATR(7)/ATR(30) < 1.5 filter avoids panic entries
- 2.0*ATR stoploss protects against 2022-style crashes
- Discrete signal levels (0.0, ±0.25) minimize fee churn

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete (max 0.35)
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_reversal_dual_htf_vol_regime_atr_v1"
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
    Better than RSI for catching reversals in bear markets.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest) / (highest - lowest) over period
    3. Transform: 0.5 * ln((1 + x) / (1 - x)) where x = 2*normalized - 1
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Typical price
    typical = (high_s + low_s) / 2
    
    # Normalize over rolling window
    highest = typical.rolling(window=period, min_periods=period).max()
    lowest = typical.rolling(window=period, min_periods=period).min()
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = range_val.replace(0, np.inf)
    
    normalized = (typical - lowest) / range_val
    
    # Transform to Fisher
    x = 2 * normalized - 1
    x = x.clip(-0.999, 0.999)  # Avoid ln(0) or ln(inf)
    
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    
    # Fisher signal line (1-period lag for trigger)
    fisher_signal = fisher.shift(1)
    
    return fisher.values, fisher_signal.values

def calculate_vol_regime(atr, short_period=7, long_period=30):
    """
    Volatility regime indicator: ATR(short) / ATR(long)
    > 1.5 = high vol (avoid entries)
    < 1.0 = low vol (good for entries)
    """
    atr_s = pd.Series(atr)
    atr_short = atr_s.ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_long = atr_s.ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_ratio = atr_short / atr_long
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0, posinf=1.0, neginf=1.0)
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    vol_ratio = calculate_vol_regime(atr_14, 7, 30)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    HALF_SIZE = 0.125
    
    # Track position state for stoploss
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === DUAL HTF TREND BIAS ===
        # Both 1d and 1w must agree for strong bias
        bull_bias_1d = close[i] > hma_1d_aligned[i]
        bull_bias_1w = close[i] > hma_1w_aligned[i]
        bear_bias_1d = close[i] < hma_1d_aligned[i]
        bear_bias_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bias: both HTF agree
        strong_bull = bull_bias_1d and bull_bias_1w
        strong_bear = bear_bias_1d and bear_bias_1w
        
        # Weak bias: only 1d agrees (1w neutral or opposite)
        weak_bull = bull_bias_1d and not strong_bull
        weak_bear = bear_bias_1d and not strong_bear
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_short = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        
        # Additional confirmation: Fisher moving in right direction
        fisher_rising = fisher[i] > fisher_signal[i] if not np.isnan(fisher_signal[i]) else False
        fisher_falling = fisher[i] < fisher_signal[i] if not np.isnan(fisher_signal[i]) else False
        
        # === VOLATILITY REGIME FILTER ===
        # Avoid entries when vol ratio > 1.5 (panic/extreme vol)
        vol_acceptable = vol_ratio[i] < 1.5
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long: Fisher reversal + strong/weak bull bias + vol acceptable
        if fisher_long and fisher_rising and vol_acceptable:
            if strong_bull:
                new_signal = SIZE  # Full position with strong bias
            elif weak_bull:
                new_signal = HALF_SIZE  # Half position with weak bias
        
        # Short: Fisher reversal + strong/weak bear bias + vol acceptable
        elif fisher_short and fisher_falling and vol_acceptable:
            if strong_bear:
                new_signal = -SIZE  # Full position with strong bias
            elif weak_bear:
                new_signal = -HALF_SIZE  # Half position with weak bias
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 1d HMA flips against position (1w is slower, use 1d for exit)
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias_1d:
                new_signal = 0.0
            if position_side < 0 and bull_bias_1d:
                new_signal = 0.0
        
        # === VOLATILITY SPIKE EXIT ===
        # Exit if vol ratio spikes > 2.0 (panic mode)
        if in_position and new_signal != 0.0:
            if vol_ratio[i] > 2.0:
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