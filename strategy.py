#!/usr/bin/env python3
"""
Experiment #627: 6h Primary + 1d/1w HTF — Fisher Transform Reversals + Vol Spike + Regime Filter

Hypothesis: 6h timeframe is underexplored (ZERO experiments). Fisher Transform excels at 
catching reversals in bear/range markets (2022 crash, 2025 bear). Combined with vol spike 
detection and HTF trend filters, this should generate quality reversal trades.

Key innovations:
1. Ehlers Fisher Transform (period=9) - proven reversal indicator for bear markets
2. Vol spike confirmation - ATR(7)/ATR(30) > 1.8 ensures we catch panic extremes
3. 1d/1w HMA trend bias - only take reversals WITH the higher timeframe trend
4. CHOP regime filter - avoid mean reversion in strong trends (CHOP < 38)
5. LOOSE Fisher thresholds (-1.2/+1.2) - ensure we generate trades (avoid 0-trade failure)

Strategy logic:
1. 1w HMA(21) = macro trend bias
2. 1d HMA(21) = primary trend filter (only long if price > 1d HMA, only short if <)
3. 6h Fisher(9) = reversal signal (cross above -1.2 = long, cross below +1.2 = short)
4. 6h ATR ratio = vol spike confirm (ratio > 1.8 = extreme move, reversal likely)
5. 6h CHOP(14) = regime filter (> 45 = range/reversion OK, < 38 = trend, skip reversion)
6. 6h ATR(14) = stoploss (2.5*ATR trailing)

Entry conditions (LOOSE to ensure trades):
- LONG: Fisher crosses above -1.2 + ATR_ratio > 1.5 + price > 1d HMA + CHOP > 40
- SHORT: Fisher crosses below +1.2 + ATR_ratio > 1.5 + price < 1d HMA + CHOP > 40

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_volspike_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_fisher(close, high, low, period=9):
    """Ehlers Fisher Transform - catches reversals in bear/range markets"""
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    prev_normalized = 0.0
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        range_val = highest - lowest
        
        if range_val < 1e-10:
            fisher[i] = 0.0
            if i > period:
                trigger[i] = fisher[i-1]
            continue
        
        # Ehlers normalization with smoothing
        normalized = 0.66 * ((close[i] - lowest) / range_val - 0.5) + 0.67 * prev_normalized
        
        # Clip to avoid division by zero in log
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher Transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        if i > period:
            trigger[i] = fisher[i-1]
        
        prev_normalized = normalized
    
    return fisher, trigger

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppy vs trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_atr_ratio(atr, short_period=7, long_period=30):
    """ATR ratio for vol spike detection"""
    n = len(atr)
    ratio = np.zeros(n)
    ratio[:] = np.nan
    
    # Calculate short and long ATR from the ATR series itself
    atr_short = pd.Series(atr).ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_long = pd.Series(atr).ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    
    for i in range(n):
        if atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMAs
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    fisher, trigger = calculate_fisher(close, high, low, period=9)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(atr, short_period=7, long_period=30)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Fisher crossover tracking
    prev_fisher = np.nan
    prev_trigger = np.nan
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            prev_trigger = trigger[i] if not np.isnan(trigger[i]) else prev_trigger
            continue
        
        if np.isnan(fisher[i]) or np.isnan(trigger[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            prev_trigger = trigger[i] if not np.isnan(trigger[i]) else prev_trigger
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            prev_trigger = trigger[i] if not np.isnan(trigger[i]) else prev_trigger
            continue
        
        if np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            prev_trigger = trigger[i] if not np.isnan(trigger[i]) else prev_trigger
            continue
        
        # === HTF BIAS (1d primary, 1w confirmation) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # 1w macro boost
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === VOL SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 1.5  # Relaxed from 1.8 to ensure trades
        
        # === CHOPPINESS REGIME ===
        # CHOP > 45 = range (mean reversion OK)
        # CHOP < 38 = strong trend (skip reversion entries)
        is_range = chop[i] > 40.0
        is_trend = chop[i] < 38.0
        
        # === FISHER CROSSOVER SIGNALS ===
        fisher_cross_long = False
        fisher_cross_short = False
        
        if not np.isnan(prev_fisher) and not np.isnan(prev_trigger):
            # Long: Fisher crosses ABOVE trigger (both rising through negative zone)
            if prev_fisher <= -1.2 and fisher[i] > -1.2:
                fisher_cross_long = True
            # Also check trigger cross for confirmation
            if prev_trigger <= -1.2 and trigger[i] > -1.2 and fisher[i] > trigger[i]:
                fisher_cross_long = True
            
            # Short: Fisher crosses BELOW trigger (both falling through positive zone)
            if prev_fisher >= 1.2 and fisher[i] < 1.2:
                fisher_cross_short = True
            if prev_trigger >= 1.2 and trigger[i] < 1.2 and fisher[i] < trigger[i]:
                fisher_cross_short = True
        
        # Also check extreme levels for direct entry
        fisher_extreme_long = fisher[i] < -1.5 and trigger[i] < -1.5
        fisher_extreme_short = fisher[i] > 1.5 and trigger[i] > 1.5
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG: HTF bull + Fisher reversal + vol spike + range regime
        if htf_bull and is_range:
            if (fisher_cross_long or fisher_extreme_long):
                if vol_spike:
                    # Strong signal with vol spike
                    if macro_bull:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
                else:
                    # Base signal without vol spike
                    desired_signal = SIZE_BASE * 0.7
        
        # SHORT: HTF bear + Fisher reversal + vol spike + range regime
        elif htf_bear and is_range:
            if (fisher_cross_short or fisher_extreme_short):
                if vol_spike:
                    # Strong signal with vol spike
                    if macro_bear:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
                else:
                    # Base signal without vol spike
                    desired_signal = -SIZE_BASE * 0.7
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.7
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
        
        # Update Fisher tracking for next iteration
        prev_fisher = fisher[i]
        prev_trigger = trigger[i]
    
    return signals