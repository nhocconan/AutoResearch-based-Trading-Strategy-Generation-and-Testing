#!/usr/bin/env python3
"""
Experiment #292: 4h Supertrend with 1d HMA Soft Bias and ATR Sizing

Hypothesis: After analyzing 291 failed experiments, the pattern is clear:
- Mean reversion on 4h is catastrophic (#290 Sharpe=-31.201)
- RSI pullback entries consistently fail (#284, #285 Sharpe~-2.5)
- Over-filtering kills trade count (#286 with 1d+1w HMA + volume = Sharpe=-0.587)
- Current best uses KAMA + 1d HMA + ADX + ATR (Sharpe=0.478)

New approach for 4h:
1. Supertrend(10, 3) as primary trend signal - cleaner than Donchian breakouts
2. 1d HMA(21) as SOFT bias (not hard filter) - weights signal but doesn't block
3. ATR-based position sizing - reduce size when volatility spikes
4. 2.5*ATR trailing stop - tighter than 12h strategies, appropriate for 4h
5. MINIMAL filters - ensure >=10 trades per symbol by not over-constraining

Key insight: The problem with #286 was TOO MANY filters (1d+1w HMA + volume + ADX).
This strategy uses ONLY 1d HMA as soft bias, allowing more trades while still
filtering the worst counter-trend entries. Supertrend provides cleaner trend
signals than Donchian breakouts on 4h timeframe.

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.35 discrete, scaled by ATR volatility
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_supertrend_1d_hma_soft_bias_atr_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=bullish, -1=bearish)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize arrays
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    # First valid bar
    supertrend[period] = upper_band[period]
    direction[period] = 1
    
    # Calculate Supertrend recursively
    for i in range(period + 1, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
            continue
            
        # If previous trend was up
        if direction[i-1] == 1:
            if close[i] > lower_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
        # If previous trend was down
        else:
            if close[i] < upper_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
    
    return supertrend, direction

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

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
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
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
        
        # === HIGHER TIMEFRAME SOFT BIAS ===
        # 1d HMA = directional bias (soft filter, not hard block)
        # We weight the signal but don't completely block counter-trend
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY ADJUSTMENT ===
        # Reduce position size when ATR is elevated (>1.5x recent average)
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Determine position size based on volatility
        if high_volatility:
            position_size = SIZE_REDUCED
        else:
            position_size = SIZE_BASE
        
        # === SUPERTREND SIGNAL ===
        # Supertrend direction: 1 = bullish (price above supertrend), -1 = bearish
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # === ENTRY CONDITIONS (LOOSE for trade count) ===
        # Long: Supertrend bullish + prefer 1d HMA bullish (but not required)
        # Short: Supertrend bearish + prefer 1d HMA bearish (but not required)
        
        new_signal = 0.0
        
        # LONG ENTRY: Supertrend bullish, 1d bias helps but not required
        if st_bullish:
            if bull_trend_1d:
                # Strong signal: both ST and 1d HMA agree
                new_signal = position_size
            else:
                # Weaker signal: ST bullish but 1d HMA bearish
                # Only enter if price > EMA50 (some trend confirmation)
                if close[i] > ema_50[i]:
                    new_signal = position_size * 0.7  # Reduced size for counter-HTF trade
        
        # SHORT ENTRY: Supertrend bearish, 1d bias helps but not required
        if st_bearish:
            if bear_trend_1d:
                # Strong signal: both ST and 1d HMA agree
                new_signal = -position_size
            else:
                # Weaker signal: ST bearish but 1d HMA bullish
                # Only enter if price < EMA50 (some trend confirmation)
                if close[i] < ema_50[i]:
                    new_signal = -position_size * 0.7  # Reduced size for counter-HTF trade
        
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
        
        # === TREND REVERSAL EXIT (Soft) ===
        # Exit if Supertrend reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and st_bearish:
                new_signal = 0.0  # Supertrend reversed against long
            if position_side < 0 and st_bullish:
                new_signal = 0.0  # Supertrend reversed against short
        
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