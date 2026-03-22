#!/usr/bin/env python3
"""
Experiment #287: 12h Supertrend with Triple HMA Regime Filter and ADX Confirmation

Hypothesis: After analyzing 286 experiments, the pattern is clear - Donchian breakouts
on 12h are too slow and generate few trades. This strategy uses:

1. 12h Supertrend(10,3) - cleaner trend signals than Donchian, more responsive
2. Triple HMA alignment (12h, 1d, 1w) - strongest regime filter (all must agree)
3. ADX(14) > 20 filter - avoid choppy/range markets where trend strategies fail
4. Volume confirmation at 1.2x (looser than 1.5x) - ensures >=10 trades per symbol
5. 2.5*ATR trailing stoploss - tighter than 3.0 for better risk control on 12h
6. Position sizing: 0.25 base, 0.35 when all 3 HTF align strongly

Why this might beat #263 (Donchian):
- Supertrend generates more signals than Donchian breakouts
- Triple HMA (12h/1d/1w) is stronger filter than single 1d HMA
- ADX filter avoids whipsaws in ranging markets (critical for 2022-2024)
- Looser volume threshold ensures we get trades without sacrificing quality
- 12h Supertrend is proven in literature for medium-term trend following

Key learnings from failures:
- #276, #282, #286: 1d Donchian had negative Sharpe (too few trades, late entries)
- #277, #284, #285: RSI pullback consistently fails on all timeframes
- #278, #281, #283: KAMA strategies had catastrophic drawdowns
- Simple trend + strong HTF bias works best (see current best: #4h KAMA + 1d HMA)

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_supertrend_triple_hma_adx_volume_v1"
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
    """Calculate Supertrend indicator."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = close[i]
            continue
        
        median = (high[i] + low[i]) / 2.0
        
        # Calculate upper and lower bands
        upper_band = median + multiplier * atr[i]
        lower_band = median - multiplier * atr[i]
        
        # Supertrend logic
        if direction[i-1] == 1:
            if close[i] < lower_band:
                direction[i] = -1
                supertrend[i] = upper_band
            else:
                direction[i] = 1
                supertrend[i] = max(lower_band, supertrend[i-1])
        else:
            if close[i] > upper_band:
                direction[i] = 1
                supertrend[i] = lower_band
            else:
                direction[i] = -1
                supertrend[i] = min(upper_band, supertrend[i-1])
    
    return supertrend, direction

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    adx = np.zeros(n)
    
    # Calculate DM and TR
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's method
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI and ADX
    for i in range(period, n):
        if tr_s[i] == 0:
            adx[i] = 0
            continue
        plus_di = 100 * plus_dm_s[i] / tr_s[i]
        minus_di = 100 * minus_dm_s[i] / tr_s[i]
        di_diff = abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        if di_sum == 0:
            adx[i] = 0
        else:
            dx = 100 * di_diff / di_sum
            adx[i] = dx  # Simplified - use DX as proxy for first calculation
    
    # Smooth ADX
    adx_s = pd.Series(adx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx_s

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    adx = calculate_adx(high, low, close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    hma_12h = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25  # Base position size
    SIZE_STRONG = 0.35  # Strong signal (all 3 HMA align)
    SIZE_WEAK = 0.20  # Weak signal (only 12h + 1d align)
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(st_direction[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === TRIPLE HMA REGIME FILTER ===
        # All three timeframes must align for strongest signals
        bull_12h = close[i] > hma_12h[i]
        bull_1d = close[i] > hma_1d_aligned[i]
        bull_1w = close[i] > hma_1w_aligned[i]
        
        bear_12h = close[i] < hma_12h[i]
        bear_1d = close[i] < hma_1d_aligned[i]
        bear_1w = close[i] < hma_1w_aligned[i]
        
        # Count bullish/bearish alignment
        bull_count = int(bull_12h) + int(bull_1d) + int(bull_1w)
        bear_count = int(bear_12h) + int(bear_1d) + int(bear_1w)
        
        # === SUPERTREND SIGNAL ===
        supertrend_bull = st_direction[i] == 1
        supertrend_bear = st_direction[i] == -1
        
        # === ADX FILTER ===
        # Only trade when ADX > 20 (trending market, not choppy)
        adx_strong = adx[i] > 20
        
        # === VOLUME CONFIRMATION ===
        # Looser threshold (1.2x) to ensure >=10 trades per symbol
        volume_confirmed = volume[i] > 1.2 * vol_sma[i]
        
        # === DETERMINE POSITION SIZE ===
        if bull_count >= 3 or bear_count >= 3:
            position_size = SIZE_STRONG  # All 3 align
        elif bull_count >= 2 or bear_count >= 2:
            position_size = SIZE_BASE  # 2 align
        else:
            position_size = SIZE_WEAK  # Only 1 align (weak signal)
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: Supertrend bull + ADX strong + volume + HMA bias
        # Require at least 2/3 HMA bullish for long
        long_conditions = (
            supertrend_bull and  # Supertrend bullish
            adx_strong and  # Trending market
            volume_confirmed and  # Volume confirms
            bull_count >= 2  # At least 2/3 HMA bullish
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            supertrend_bear and  # Supertrend bearish
            adx_strong and  # Trending market
            volume_confirmed and  # Volume confirms
            bear_count >= 2  # At least 2/3 HMA bearish
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
        # Exit if Supertrend reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and supertrend_bear:
                new_signal = 0.0  # Supertrend reversed against long
            if position_side < 0 and supertrend_bull:
                new_signal = 0.0  # Supertrend reversed against short
        
        # === EXIT IF REGIME CHANGES ===
        # Exit long if bearish regime takes over (3/3 bearish)
        if in_position and position_side > 0 and bear_count >= 3:
            new_signal = 0.0
        
        # Exit short if bullish regime takes over (3/3 bullish)
        if in_position and position_side < 0 and bull_count >= 3:
            new_signal = 0.0
        
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
            # else: maintaining same position direction (possibly adjusted size)
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