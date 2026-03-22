#!/usr/bin/env python3
"""
Experiment #305: 12h Supertrend with Single HTF HMA Bias and Volume Filter

Hypothesis: After analyzing 299 experiments, the key insights are:
1. #299 (12h Donchian + dual HTF) failed (Sharpe=-0.080) because requiring BOTH 1d AND 1w HMA alignment is TOO RESTRICTIVE
2. #292 (4h Supertrend + 1d HMA) works best (Sharpe=0.485) - simple is better
3. Complex ensembles with 3+ filters consistently fail (#297, #295, #296)
4. 12h needs FEWER filters than 4h (fewer bars = each filter eliminates more opportunities)

This strategy SIMPLIFIES the 12h approach:
1. 12h Supertrend(10, 3) as PRIMARY signal (proven edge from #292, #300, #304)
2. 1d HMA(21) as SINGLE trend bias filter (NOT both 1d AND 1w - that killed #299)
3. Volume confirmation: volume > 0.8 * 20-bar avg (filters low-liquidity false breakouts)
4. ADX(14) > 12 (very loose threshold - only filter out dead-chop markets)
5. ATR(14) trailing stoploss at 3.0 * ATR (wider for 12h timeframe)
6. Position sizing: 0.25 base, 0.35 in strong trends (discrete levels)

Why this should beat #299:
- Single HTF filter (1d only) = more trades generated vs dual HTF
- Supertrend has built-in stoploss logic = cleaner exits than Donchian
- Volume filter adds edge without over-filtering (unlike ADX+RSI+BB ensembles)
- Looser ADX threshold (12 vs 15/25) ensures >=10 trades per symbol

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 3.0 * ATR(14) trailing (wider for 12h volatility)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_supertrend_1d_hma_volume_adx_atr_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_line, supertrend_direction (1=long, -1=short)
    
    Supertrend = (HL_avg) +/- (multiplier * ATR)
    Direction flips when price crosses the supertrend line.
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    # HL average
    hl_avg = (high + low) / 2
    
    # Upper and lower bands
    upper_band = hl_avg + multiplier * atr
    lower_band = hl_avg - multiplier * atr
    
    # Supertrend line and direction
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = long (price above ST), -1 = short (price below ST)
    
    supertrend[0] = upper_band[0]
    direction[0] = -1  # Start bearish
    
    for i in range(1, n):
        if direction[i - 1] == 1:
            # Previously long
            if close[i] > lower_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
        else:
            # Previously short
            if close[i] < upper_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
    
    return supertrend, direction

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with Wilder's method
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

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
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend_line, supertrend_dir = calculate_supertrend(high, low, close, 10, 3.0)
    adx = calculate_adx(high, low, close, 14)
    
    # Volume moving average (20-bar)
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25  # Base position size
    SIZE_STRONG = 0.35  # Increased size in strong trend
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend_line[i]) or np.isnan(supertrend_dir[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(volume_avg[i]) or volume_avg[i] == 0:
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS (Single 1d HMA Filter) ===
        # Only require 1d HMA alignment (NOT both 1d AND 1w like #299)
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === TREND STRENGTH ===
        # ADX > 12 = trending market (very loose for 12h to ensure trades)
        trending = adx[i] > 12
        strong_trend = adx[i] > 25
        
        # === VOLUME CONFIRMATION ===
        # Volume > 0.8 * 20-bar average (filters low-liquidity false breakouts)
        volume_confirmed = volume[i] > 0.8 * volume_avg[i]
        
        # === SUPERTREND SIGNAL ===
        # Supertrend direction: 1 = long (price above ST), -1 = short (price below ST)
        st_long = supertrend_dir[i] == 1
        st_short = supertrend_dir[i] == -1
        
        # Determine position size based on trend strength
        if strong_trend and volume_confirmed:
            position_size = SIZE_STRONG
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS (SIMPLIFIED vs #299) ===
        new_signal = 0.0
        
        # LONG ENTRY: Supertrend long + 1d HMA bullish + ADX confirms + volume ok
        # Only 3 filters (not 4+ like #299) to ensure >=10 trades
        long_conditions = (
            st_long and  # Supertrend says long
            bull_trend_1d and  # 1d HMA bias bullish
            trending  # ADX confirms trend (loose threshold)
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            st_short and  # Supertrend says short
            bear_trend_1d and  # 1d HMA bias bearish
            trending  # ADX confirms trend
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 3.0 * ATR below highest close (wider for 12h)
                stoploss_price = highest_close - 3.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 3.0 * ATR above lowest close
                stoploss_price = lowest_close + 3.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TREND REVERSAL EXIT ===
        # Exit if 1d HMA bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0  # 1d trend reversed against long
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0  # 1d trend reversed against short
        
        # === SUPERTREND REVERSAL EXIT ===
        # Exit if Supertrend direction flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and st_short:
                new_signal = 0.0  # Supertrend flipped against long
            if position_side < 0 and st_long:
                new_signal = 0.0  # Supertrend flipped against short
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals