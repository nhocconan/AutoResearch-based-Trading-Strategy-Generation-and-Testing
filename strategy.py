#!/usr/bin/env python3
"""
Experiment #167: 12h Supertrend + 1d HMA Trend Bias + Choppiness Filter

Hypothesis: 12h Supertrend provides clear trend-following signals with built-in
volatility adjustment. 1d HMA gives higher timeframe trend bias to avoid
counter-trend trades. Choppiness Index filters out ranging markets where
Supertrend whipsaws. This combo should beat pure Donchian breakout (#161).

Why this might work better than #161:
- Supertrend adapts to volatility (ATR-based), Donchian is fixed period
- Choppiness filter prevents trades in sideways markets (major Sharpe killer)
- 12h is slow enough to capture sustained trends, fast enough for 10+ trades/year
- Learning from #161: ADX>20 was too restrictive, CHOP<50 is more adaptive

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: Built into Supertrend + 2.5*ATR emergency stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_supertrend_1d_hma_chop_atr_v1"
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
    """Calculate Supertrend indicator for trend direction."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    # Calculate basic bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize final bands
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    
    # First bar
    final_upper[0] = upper_band[0]
    final_lower[0] = lower_band[0]
    supertrend[0] = upper_band[0]  # Start in short mode
    
    # Iterate for subsequent bars
    for i in range(1, n):
        # Upper band logic
        if upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = final_upper[i-1]
        
        # Lower band logic
        if lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = final_lower[i-1]
        
        # Determine Supertrend value
        if supertrend[i-1] == final_upper[i-1]:
            if close[i] > final_upper[i]:
                supertrend[i] = final_lower[i]
            else:
                supertrend[i] = final_upper[i]
        else:
            if close[i] < final_lower[i]:
                supertrend[i] = final_upper[i]
            else:
                supertrend[i] = final_lower[i]
    
    # Determine trend direction: 1 = long (price > supertrend), -1 = short
    trend = np.where(close > supertrend, 1, -1)
    
    return supertrend, trend, atr

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP) for regime detection.
    
    CHOP > 61.8 = ranging/choppy market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        atr_sum = 0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], 
                     abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                     abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    supertrend, st_trend, atr = calculate_supertrend(high, low, close, 10, 3.0)
    chop = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
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
        
        if np.isnan(chop[i]) or np.isnan(st_trend[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS FILTER ===
        # CHOP < 50 = trending market (trade Supertrend signals)
        # CHOP >= 50 = ranging market (reduce position or stay flat)
        is_trending = chop[i] < 50
        
        # === SUPERTREND SIGNAL ===
        # st_trend = 1 means price above Supertrend (bullish)
        # st_trend = -1 means price below Supertrend (bearish)
        st_long = st_trend[i] == 1
        st_short = st_trend[i] == -1
        
        # Check for Supertrend flip (entry signal)
        st_flip_long = (i > 0 and st_trend[i] == 1 and st_trend[i-1] == -1)
        st_flip_short = (i > 0 and st_trend[i] == -1 and st_trend[i-1] == 1)
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # 1d bullish + trending market + Supertrend flip to long
        if bull_trend_1d and is_trending and st_flip_long:
            new_signal = SIZE_STRONG
        elif bull_trend_1d and is_trending and st_long:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # 1d bearish + trending market + Supertrend flip to short
        if bear_trend_1d and is_trending and st_flip_short:
            new_signal = -SIZE_STRONG
        elif bear_trend_1d and is_trending and st_short:
            new_signal = -SIZE_BASE
        
        # === RANGING MARKET ADJUSTMENT ===
        # In choppy markets, reduce position size by half
        if not is_trending and new_signal != 0.0:
            new_signal = new_signal * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - Supertrend + 2.5*ATR emergency ===
        # Supertrend itself acts as trailing stop, but add emergency stop
        
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Emergency stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            # Also check Supertrend level
            st_stop = supertrend[i]
            if close[i] < stoploss_price or close[i] < st_stop:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Emergency stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            # Also check Supertrend level
            st_stop = supertrend[i]
            if close[i] > stoploss_price or close[i] > st_stop:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals