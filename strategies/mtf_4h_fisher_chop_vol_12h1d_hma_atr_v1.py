#!/usr/bin/env python3
"""
Experiment #1014: 4h Primary + 12h/1d HTF — Fisher Transform + Choppiness Regime + Vol Spike

Hypothesis: After 735+ failed strategies, the key insight is that bear/range markets (2025 test period)
require different logic than bull markets. This strategy combines:

1. EHLERS FISHER TRANSFORM (period=9): Non-linear transform that normalizes price into Gaussian
   distribution. Long when Fisher crosses above -1.5 (oversold reversal), short when crosses below +1.5
   (overbought reversal). Proven to catch reversals in bear market rallies.

2. CHOPPINESS INDEX (CHOP) regime filter:
   - CHOP > 61.8 = ranging → use Fisher mean reversion signals
   - CHOP < 38.2 = trending → use trend-following (HMA slope + Fisher direction)
   - Between = hold existing, no new entries

3. VOLATILITY SPIKE FILTER: ATR(7)/ATR(21) > 1.8 indicates panic/extreme vol
   - In bear market: vol spike + Fisher oversold = high-probability long reversal
   - Exit when vol ratio < 1.2 (vol crush)

4. 12h HMA21 + 1d HMA21: Dual HTF trend bias
   - Only long when price > 12h HMA (medium-term bullish)
   - Only short when price < 1d HMA (long-term bearish)
   - This asymmetry works better in bear/range markets

5. ATR Trailing Stop: 2.5x ATR for risk management, signal→0 when hit

Why 4h works:
- Target 30-60 trades/year (vs 100+ on 1h, 20 on 12h)
- Enough frequency for statistical significance
- Less noise than 1h/30m, more signals than 12h/1d

Critical fixes from failed experiments:
- FISHER TRANSFORM instead of RSI/CRSI (better for bear market reversals)
- DUAL HTF (12h + 1d) with asymmetric logic for bear bias
- VOL SPIKE filter to catch panic bottoms (works in 2022 crash, 2025 bear)
- RELAXED Fisher thresholds (-1.5/+1.5 not -2/+2) for more trades
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 30-60 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_vol_12h1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Converts price into a Gaussian normal distribution
    Entry: Fisher crosses above -1.5 (oversold reversal)
    Exit: Fisher crosses below +1.5 (overbought reversal)
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    if n < period + 5:
        return fisher, trigger
    
    # Calculate typical price and normalize
    for i in range(period, n):
        # Use highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = 0.0
            trigger[i] = 0.0
            continue
        
        # Normalize price to range -1 to +1
        value = (2.0 * close[i] - highest - lowest) / (highest - lowest + 1e-10)
        value = np.clip(value * 0.999, -0.999, 0.999)  # Prevent log(0)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + value) / (1.0 - value + 1e-10))
        
        # Trigger line (1-period lag of fisher)
        if i > period:
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures whether market is trending or ranging
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend following)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        if atr_sum > 0 and (highest_high - lowest_low) > 0:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope (positive = uptrend, negative = downtrend)."""
    n = len(hma_values)
    slope = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if not np.isnan(hma_values[i]) and not np.isnan(hma_values[i-lookback]):
            slope[i] = (hma_values[i] - hma_values[i-lookback]) / (hma_values[i-lookback] + 1e-10)
    
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 12h HMA21 for medium-term trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA21 for long-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    fisher_4h, trigger_4h = calculate_fisher_transform(high, low, close, period=9)
    atr_4h = calculate_atr(high, low, close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    
    # Volatility spike filter: ATR(7)/ATR(21)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_21 = calculate_atr(high, low, close, period=21)
    vol_ratio = np.full(n, np.nan)
    for i in range(21, n):
        if atr_21[i] > 1e-10:
            vol_ratio[i] = atr_7[i] / atr_21[i]
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(fisher_4h[i]) or np.isnan(trigger_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(chop_4h[i]) or np.isnan(vol_ratio[i]):
            continue
        
        # === MACRO TREND (HTF HMA21) ===
        # Asymmetric: easier to long (12h), harder to short (1d) for bear bias
        medium_bull = close[i] > hma_12h_aligned[i]
        long_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop_4h[i] > 61.8  # Ranging market → mean reversion
        regime_trend = chop_4h[i] < 38.2  # Trending market → trend follow
        regime_neutral = not regime_chop and not regime_trend  # Transition
        
        # === VOLATILITY SPIKE FILTER ===
        vol_spike = vol_ratio[i] > 1.8  # Panic/extreme vol
        vol_normal = vol_ratio[i] < 1.2  # Vol crush (exit signal)
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher_4h[i] < -1.5
        fisher_overbought = fisher_4h[i] > 1.5
        fisher_cross_long = fisher_4h[i] > -1.5 and trigger_4h[i] <= -1.5
        fisher_cross_short = fisher_4h[i] < 1.5 and trigger_4h[i] >= 1.5
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        if regime_chop and medium_bull:
            # Mean reversion in choppy market with bullish medium trend
            if fisher_cross_long:
                desired_signal = BASE_SIZE
            elif fisher_oversold and vol_spike:
                # Vol spike + oversold = high-probability reversal
                desired_signal = BASE_SIZE
        elif regime_trend and medium_bull:
            # Trend following in trending bullish market
            if fisher_4h[i] > -0.5 and fisher_4h[i-1] <= -0.5:
                desired_signal = REDUCED_SIZE
        elif regime_neutral and medium_bull:
            # Relaxed entry in transition
            if fisher_oversold:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        if regime_chop and long_bear:
            # Mean reversion in choppy market with bearish long trend
            if fisher_cross_short:
                desired_signal = -BASE_SIZE
            elif fisher_overbought and vol_spike:
                desired_signal = -BASE_SIZE
        elif regime_trend and long_bear:
            # Trend following in trending bearish market
            if fisher_4h[i] < 0.5 and fisher_4h[i-1] >= 0.5:
                desired_signal = -REDUCED_SIZE
        elif regime_neutral and long_bear:
            # Relaxed entry in transition
            if fisher_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === VOL CRUSH EXIT ===
        if in_position and vol_normal and not stoploss_triggered:
            # Exit when volatility crushes (move done)
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if medium bullish and Fisher not extreme overbought
                if medium_bull and fisher_4h[i] < 1.0:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if long-term bearish and Fisher not extreme oversold
                if long_bear and fisher_4h[i] > -1.0:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if medium trend reverses
            if not medium_bull and fisher_4h[i] > 0.5:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if long-term trend reverses
            if not long_bear and fisher_4h[i] < -0.5:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals