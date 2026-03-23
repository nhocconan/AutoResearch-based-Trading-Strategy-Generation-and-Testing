#!/usr/bin/env python3
"""
Experiment #300: 1h Primary + 4h/12h HTF — Fisher Transform Regime Strategy

Hypothesis: Previous 1h strategies (#290, #295, #298) failed due to over-filtering
(session/volume filters killed all trades). This version removes those filters and
uses Ehlers Fisher Transform for entry timing - proven to catch reversals in
bear/range markets better than RSI.

KEY CHANGES from failed 1h experiments:
- NO session filter (killed trades in #295, #298)
- NO volume filter (killed trades in #290)
- Fisher Transform instead of RSI (better for bear market reversals)
- Looser regime thresholds (CHOP 50/60 vs 45/55) to get more trend time
- ATR stop at 2.0x (vs 2.5x) to reduce premature exits
- Position size 0.25 (conservative for 1h TF)

ENTRY LOGIC:
- 12h HMA(21) = MACRO bias (only long if price > 12h HMA, only short if <)
- 4h HMA(21) = INTERMEDIATE trend (confirms direction)
- Fisher Transform(9) = ENTRY timing (crosses -1.5 long, +1.5 short)
- Choppiness(14) = REGIME filter (>60 range, <50 trend)
- ATR(14) 2.0x trailing stoploss

TARGET: 40-70 trades/year on 1h, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_regime_hma_4h12h_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    hl2 = (high + low) / 2
    hl2_s = pd.Series(hl2)
    
    # Calculate highest high and lowest low over period
    highest = hl2_s.rolling(window=period, min_periods=period).max()
    lowest = hl2_s.rolling(window=period, min_periods=period).min()
    
    # Normalize price to 0-1 range
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = (hl2 - lowest) / (highest - lowest + 1e-10)
    
    # Clamp to 0.001-0.999 to avoid log(0)
    normalized = np.clip(normalized, 0.001, 0.999)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
    fisher_prev = np.roll(fisher, 1)
    fisher_prev[0] = fisher[0]
    
    return fisher.values, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range, CHOP < 38.2 = trend
    Using 50/60 thresholds for this strategy.
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.25  # Conservative for 1h timeframe
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (12h HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 60 = range (mean revert), CHOP < 50 = trend (follow)
        is_choppy = chop[i] > 60.0
        is_trending = chop[i] < 50.0
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_long = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_cross_short = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_choppy:
            # RANGE REGIME: Mean reversion with stricter bias
            # Long: Fisher oversold + price above 12h HMA (bullish bias in range)
            if fisher_cross_long and price_above_hma_12h:
                desired_signal = POSITION_SIZE
            # Short: Fisher overbought + price below 12h HMA (bearish bias in range)
            elif fisher_cross_short and price_below_hma_12h:
                desired_signal = -POSITION_SIZE
        
        else:  # is_trending or neutral (50-60)
            # TREND REGIME: Follow intermediate trend with Fisher entry
            # Long: Fisher oversold + price above 4h HMA (pullback in uptrend)
            if fisher_cross_long and price_above_hma_4h:
                desired_signal = POSITION_SIZE
            # Short: Fisher overbought + price below 4h HMA (rally in downtrend)
            elif fisher_cross_short and price_below_hma_4h:
                desired_signal = -POSITION_SIZE
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === MACRO BIAS REVERSAL EXIT ===
        # Exit long if price crosses below 12h HMA (macro trend broken)
        if in_position and position_side > 0 and price_below_hma_12h:
            desired_signal = 0.0
        
        # Exit short if price crosses above 12h HMA (macro trend broken)
        if in_position and position_side < 0 and price_above_hma_12h:
            desired_signal = 0.0
        
        # === FISHER EXTREME EXIT (take profit) ===
        # Exit long when Fisher reaches overbought (>1.0)
        if in_position and position_side > 0 and fisher[i] > 1.0:
            desired_signal = 0.0
        
        # Exit short when Fisher reaches oversold (<-1.0)
        if in_position and position_side < 0 and fisher[i] < -1.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Maintain position if macro bias still supports
            if position_side > 0 and price_above_hma_12h:
                desired_signal = POSITION_SIZE
            elif position_side < 0 and price_below_hma_12h:
                desired_signal = -POSITION_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals