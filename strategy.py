#!/usr/bin/env python3
"""
Experiment #269: 12h Supertrend with 1d HMA Bias and ADX Filter

Hypothesis: After analyzing 268 experiments, the pattern shows:
- RSI mean reversion FAILS on crypto (Sharpe -1.6 to -3.1)
- Simple trend + HTF bias WORKS (current best Sharpe=0.478)
- Donchian breakout on 12h was CLOSE (#263 Sharpe=-0.076)

This strategy improves on #263 by:
1. Supertrend(ATR=10, mult=3) instead of Donchian - adapts to volatility better
2. ADX(14) > 20 filter - only trade when trend has strength (avoid chop)
3. 1d HMA(21) bias - proven HTF filter from current best strategy
4. 2.5*ATR trailing stop - tighter than #263's 3.0*ATR for better R:R
5. Asymmetric entries - only long when 1d HMA bullish, only short when bearish

Why 12h Supertrend might beat 4h KAMA:
- Fewer false signals due to longer timeframe
- Supertrend inherently includes ATR volatility adjustment
- 1d HMA bias is stronger at 12h than 4h (more separation)
- ADX filter prevents whipsaw entries in ranging markets

Key differences from failed strategies:
- NO RSI (consistently failed - see #251, #254, #257, #259)
- NO complex ensemble/voting ( #256 failed with Sharpe=-0.231)
- NO Choppiness Index ( #262 didn't help, Sharpe=-0.159)
- Simple: Supertrend + ADX + HTF bias + ATR stop

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete, scaled by ADX strength
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_supertrend_1d_hma_adx_atr_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=bullish, -1=bearish)
    """
    atr = calculate_atr(high, low, close, period)
    n = len(close)
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize supertrend values
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = bullish (price above ST), -1 = bearish
    
    supertrend[0] = upper_band[0]
    direction[0] = -1  # Start bearish
    
    for i in range(1, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
            continue
        
        # If previous supertrend was below price (bullish)
        if direction[i-1] == 1:
            # New lower band can only move up
            if lower_band[i] < supertrend[i-1]:
                supertrend[i] = supertrend[i-1]
            else:
                supertrend[i] = lower_band[i]
            
            # Check if price breaks below supertrend
            if close[i] < supertrend[i]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
        else:
            # Previous supertrend was above price (bearish)
            # New upper band can only move down
            if upper_band[i] > supertrend[i-1]:
                supertrend[i] = supertrend[i-1]
            else:
                supertrend[i] = upper_band[i]
            
            # Check if price breaks above supertrend
            if close[i] > supertrend[i]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
    
    return supertrend, direction

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 indicates strong trend, ADX < 20 indicates range.
    """
    n = len(close)
    
    # Calculate True Range and Directional Movement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_diff = high[i] - high[i-1]
        minus_diff = low[i-1] - low[i]
        
        if plus_diff > minus_diff and plus_diff > 0:
            plus_dm[i] = plus_diff
        if minus_diff > plus_diff and minus_diff > 0:
            minus_dm[i] = minus_diff
    
    # Smooth using Wilder's method (EMA with span=period)
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_s[i] > 0:
            plus_di[i] = 100 * plus_dm_s[i] / tr_s[i]
            minus_di[i] = 100 * minus_dm_s[i] / tr_s[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Smooth DX to get ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30  # Base position size
    SIZE_STRONG = 0.35  # Strong trend (ADX > 30)
    SIZE_WEAK = 0.20  # Weak trend (ADX 20-25)
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = strong directional bias (hard filter)
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === SUPERTREND SIGNAL ===
        # st_direction = 1 means bullish (price above supertrend)
        # st_direction = -1 means bearish (price below supertrend)
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # === ADX TREND STRENGTH FILTER ===
        # Only trade when ADX indicates trend strength
        adx_strong = adx[i] > 25  # Strong trend
        adx_moderate = adx[i] > 20  # Moderate trend
        
        # === DETERMINE POSITION SIZE BASED ON ADX ===
        if adx[i] > 30:
            position_size = SIZE_STRONG
        elif adx[i] > 25:
            position_size = SIZE_BASE
        elif adx[i] > 20:
            position_size = SIZE_WEAK
        else:
            position_size = 0.0  # No trade in range
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: Need 1d bias up + Supertrend bullish + ADX confirms trend
        # Asymmetric: only long when 1d HMA is bullish
        if bull_trend_1d and st_bullish and adx_moderate and position_size > 0:
            new_signal = position_size
        
        # SHORT ENTRY: Mirror of long
        # Asymmetric: only short when 1d HMA is bearish
        if bear_trend_1d and st_bearish and adx_moderate and position_size > 0:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TREND REVERSAL EXIT ===
        # Exit if HTF bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0  # 1d trend reversed against long
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0  # 1d trend reversed against short
        
        # Exit if Supertrend reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and st_bearish:
                new_signal = 0.0  # Supertrend flipped bearish
            if position_side < 0 and st_bullish:
                new_signal = 0.0  # Supertrend flipped bullish
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals