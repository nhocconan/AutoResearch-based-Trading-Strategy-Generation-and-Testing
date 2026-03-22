#!/usr/bin/env python3
"""
Experiment #130: 4h Fisher Transform + 1d HMA Trend + Choppiness Regime + ATR Stop

Hypothesis: After 129 failed experiments, most trend-following strategies fail on BTC/ETH
due to 2022 crash whipsaw and 2025 bear market. This strategy uses:
- Fisher Transform (period=9): Catches reversals in bear market rallies (proven edge)
- 1d HMA(21): Higher timeframe trend bias from mtf_data helper
- Choppiness Index (14): Filters regime - CHOP>61.8=range, CHOP<38.2=trend
- ATR(14) 2.5x trailing stop: Protects capital during crashes
- Asymmetric logic: Different entry rules for bull vs bear regimes

Why this might work where others failed:
- Fisher Transform excels at catching tops/bottoms in volatile crypto
- Choppiness filter avoids trend-following losses in choppy 2022 period
- 1d HMA provides stable bias without whipsaw of faster MTF
- 4h timeframe balances signal frequency vs noise (not too fast, not too slow)
- Position sizing 0.20-0.35 limits drawdown during 77% BTC crash

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_1d_hma_chop_regime_atr_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform.
    Converts price to Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    if n < period:
        return fisher, fisher_signal
    
    # Calculate typical price and normalize
    hl2 = (high + low) / 2
    hh = pd.Series(hl2).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(hl2).rolling(window=period, min_periods=period).min().values
    
    # Normalize to -1 to +1 range
    norm = np.zeros(n)
    norm[:] = np.nan
    for i in range(period, n):
        if hh[i] > ll[i]:
            norm[i] = 2 * ((hl2[i] - ll[i]) / (hh[i] - ll[i])) - 1
        else:
            norm[i] = 0
        # Clamp to avoid division errors
        norm[i] = np.clip(norm[i], -0.999, 0.999)
    
    # Fisher Transform
    fisher_vals = np.zeros(n)
    fisher_vals[:] = np.nan
    for i in range(period, n):
        if np.abs(norm[i]) < 0.999:
            fisher_vals[i] = 0.5 * np.log((1 + norm[i]) / (1 - norm[i]))
    
    # Fisher signal (1-period lag of fisher)
    fisher_signal = np.roll(fisher_vals, 1)
    fisher_signal[:period+1] = np.nan
    
    return fisher_vals, fisher_signal

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    Measures market choppiness vs trending.
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    if n < period:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            atr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], 
                        abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                        abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j])
                atr_sum += tr
            
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 100
    
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
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    chop = calculate_choppiness_index(high, low, close, 14)
    
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
    
    # Track Fisher crosses
    prev_fisher = np.nan
    prev_fisher_signal = np.nan
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME FILTER ===
        # CHOP > 61.8 = ranging (prefer mean reversion)
        # CHOP < 38.2 = trending (prefer trend following)
        # CHOP 38.2-61.8 = transitional
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 from below = bullish reversal
        # Fisher crosses below +1.5 from above = bearish reversal
        fisher_bull_cross = False
        fisher_bear_cross = False
        
        if i > 0 and not np.isnan(fisher_signal[i-1]) and not np.isnan(fisher[i-1]):
            # Bullish: Fisher was below -1.5, now crossing above
            if fisher_signal[i-1] < -1.5 and fisher[i] > -1.5:
                fisher_bull_cross = True
            # Bearish: Fisher was above +1.5, now crossing below
            if fisher_signal[i-1] > 1.5 and fisher[i] < 1.5:
                fisher_bear_cross = True
        
        # Also check extreme levels for mean reversion
        fisher_oversold = fisher[i] < -2.0
        fisher_overbought = fisher[i] > 2.0
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Strong: 1d bullish + trending regime + Fisher bull cross
        if bull_trend_1d and is_trending and fisher_bull_cross:
            new_signal = SIZE_STRONG
        # Moderate: 1d bullish + Fisher bull cross (any regime)
        elif bull_trend_1d and fisher_bull_cross:
            new_signal = SIZE_BASE
        # Range market mean reversion: 1d bullish + ranging + Fisher oversold
        elif bull_trend_1d and is_ranging and fisher_oversold:
            new_signal = SIZE_BASE
        # Ensure trades: 1d bullish + Fisher turning up
        elif bull_trend_1d and fisher[i] > fisher_signal[i] and fisher[i] < 0:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Strong: 1d bearish + trending regime + Fisher bear cross
        if bear_trend_1d and is_trending and fisher_bear_cross:
            new_signal = -SIZE_STRONG
        # Moderate: 1d bearish + Fisher bear cross (any regime)
        elif bear_trend_1d and fisher_bear_cross:
            new_signal = -SIZE_BASE
        # Range market mean reversion: 1d bearish + ranging + Fisher overbought
        elif bear_trend_1d and is_ranging and fisher_overbought:
            new_signal = -SIZE_BASE
        # Ensure trades: 1d bearish + Fisher turning down
        elif bear_trend_1d and fisher[i] < fisher_signal[i] and fisher[i] > 0:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
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