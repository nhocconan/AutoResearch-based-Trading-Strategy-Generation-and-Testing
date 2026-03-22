#!/usr/bin/env python3
"""
Experiment #268: 4h Supertrend with 1d HMA Bias and Volume Confirmation

Hypothesis: After analyzing 267 experiments, clear patterns emerge:
1. RSI mean-reversion consistently FAILS on BTC/ETH (Sharpe -1.6 to -3.1)
2. Complex ensembles fail due to fee churn and conflicting signals
3. Simple trend + strong HTF bias + proper risk management WORKS
4. Current best (mtf_4h_kama_1d_hma_adx_atr_v1) has Sharpe=0.478

This strategy uses:
1. 4h Supertrend(10, 3.0) - ATR-based trend with built-in volatility adaptation
2. 1d HMA(21) - Strong directional bias filter (prevents counter-trend trades)
3. Volume confirmation - Breakout must have volume > 1.3x 20-period average
4. ATR-based position sizing - Reduce size when volatility is elevated
5. 2.5*ATR trailing stoploss - Tighter than 12h strategies, appropriate for 4h
6. Asymmetric entries - Only trade in direction of 1d HMA bias

Why 4h Supertrend might work:
- Supertrend adapts to volatility automatically (unlike fixed Donchian)
- 4h captures medium-term trends without intraday noise
- 1d HMA bias is proven filter (works in current best strategy)
- Volume confirmation reduces false breakouts
- Fewer trades than 1h/15m = less fee drag

Key differences from failed strategies:
- NO RSI (RSI strategies consistently failed - see #251, #254, #257, #259)
- NO Choppiness Index (didn't help in #262)
- NO complex voting/ensemble (#256 ensemble had Sharpe=-0.231)
- Simple Supertrend + volume + HTF bias = cleaner signals
- Looser entry thresholds to ensure >=10 trades per symbol

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.35 discrete, scaled by ATR
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_supertrend_1d_hma_volume_atr_v1"
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

def calculate_supertrend(high, low, close, atr, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=long, -1=short)
    
    Supertrend Upper = (High + Low) / 2 + multiplier * ATR
    Supertrend Lower = (High + Low) / 2 - multiplier * ATR
    
    Direction changes when price crosses the Supertrend line.
    """
    n = len(close)
    hl2 = (high + low) / 2
    
    supertrend_upper = hl2 + multiplier * atr
    supertrend_lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = long (price above ST), -1 = short (price below ST)
    
    # Initialize
    supertrend[0] = supertrend_upper[0]
    direction[0] = -1 if close[0] < supertrend[0] else 1
    
    for i in range(1, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
            continue
        
        # If trend is currently long (price above supertrend)
        if direction[i-1] == 1:
            # Supertrend can only move up (support level)
            supertrend[i] = max(supertrend_lower[i], supertrend[i-1])
            # Check if price broke below supertrend
            if close[i] < supertrend[i]:
                direction[i] = -1
                supertrend[i] = supertrend_upper[i]
            else:
                direction[i] = 1
        else:
            # Trend is currently short (price below supertrend)
            # Supertrend can only move down (resistance level)
            supertrend[i] = min(supertrend_upper[i], supertrend[i-1])
            # Check if price broke above supertrend
            if close[i] > supertrend[i]:
                direction[i] = 1
                supertrend[i] = supertrend_lower[i]
            else:
                direction[i] = -1
    
    return supertrend, direction

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, atr, 3.0)
    vol_sma = calculate_volume_sma(volume, 20)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30  # Base position size
    SIZE_REDUCED = 0.20  # Reduced size in high vol
    SIZE_MAX = 0.35  # Maximum position size
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
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
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = strong directional bias (hard filter)
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        # Breakout must have volume > 1.3x average to be valid
        volume_confirmed = volume[i] > 1.3 * vol_sma[i]
        
        # === VOLATILITY ADJUSTMENT ===
        # Reduce position size when ATR is elevated (>1.5x recent average)
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Determine position size based on volatility
        if high_volatility:
            position_size = SIZE_REDUCED
        else:
            position_size = SIZE_BASE
        
        # === SUPERTREND SIGNALS ===
        # Supertrend direction: 1 = long (price above ST), -1 = short (price below ST)
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: Need 1d bias up + Supertrend long + volume confirmation
        # Looser conditions to ensure >=10 trades per symbol
        long_conditions = (
            bull_trend_1d and  # 1d HMA bias bullish
            st_long and  # Supertrend says long
            volume_confirmed  # Volume confirms
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            bear_trend_1d and  # 1d HMA bias bearish
            st_short and  # Supertrend says short
            volume_confirmed  # Volume confirms
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
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
        
        # === SUPERTREND REVERSAL EXIT ===
        # Exit if Supertrend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and st_short:
                new_signal = 0.0  # Supertrend flipped to short
            if position_side < 0 and st_long:
                new_signal = 0.0  # Supertrend flipped to long
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly adjusted size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals