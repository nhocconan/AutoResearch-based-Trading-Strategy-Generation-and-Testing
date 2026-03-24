#!/usr/bin/env python3
"""
Experiment #863: 6h Primary + 1d/1w HTF — Fisher Transform Reversals + HMA Trend Bias

Hypothesis: 6h timeframe with daily/weekly HTF bias captures medium-term swings while
avoiding noise. Ehlers Fisher Transform excels at reversal detection in bear/range markets
(2025 test period). Combined with 1d HMA trend bias and 1w major trend filter, this should
work better than pure trend-following which failed in 2022 crash and 2025 bear.

Key innovations:
1. 1w HMA(21) for major trend direction (bull/bear market filter)
2. 1d HMA(21) for intermediate trend bias
3. 6h Fisher Transform(9) for reversal entries (turning points)
4. Choppiness Index(14) regime: <45 trend-follow, >55 mean-revert
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.20, ±0.30

Entry conditions (LOOSE to ensure ≥10 trades/train, ≥3/test):
- LONG: 1w HMA bull + 1d HMA bull + Fisher < -1.5 (oversold reversal)
- SHORT: 1w HMA bear + 1d HMA bear + Fisher > +1.5 (overbought reversal)
- RANGE: CHOP>55 + Fisher extremes without HTF filter

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_hma_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.zeros(n)
    diff[:] = np.nan
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian distribution for clearer reversal signals
    Fisher crosses above -1.5 = long signal
    Fisher crosses below +1.5 = short signal
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Calculate typical price
    typical = (high + low) / 2.0
    
    # Normalize to -1 to +1 range
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(typical[i - period + 1:i + 1])
        lowest = np.min(typical[i - period + 1:i + 1])
        range_val = highest - lowest
        
        if range_val > 1e-10:
            normalized = 0.9999 * (typical[i] - lowest) / range_val + 0.0001
            if normalized > 0.999:
                normalized = 0.999
            elif normalized < 0.001:
                normalized = 0.001
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
            
            # Trigger line (previous fisher)
            if i > 0 and not np.isnan(fisher[i-1]):
                trigger[i] = fisher[i-1]
            else:
                trigger[i] = fisher[i]
    
    return fisher, trigger

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Using 45/55 thresholds for regime detection
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    fisher, trigger = calculate_fisher_transform(high, low, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(trigger[i]) or np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w and 1d HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crossing above trigger from below = long
        # Fisher crossing below trigger from above = short
        fisher_long_cross = False
        fisher_short_cross = False
        
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(trigger[i-1]):
            # Long: Fisher was below trigger, now above
            fisher_long_cross = (fisher[i-1] <= trigger[i-1]) and (fisher[i] > trigger[i])
            # Short: Fisher was above trigger, now below
            fisher_short_cross = (fisher[i-1] >= trigger[i-1]) and (fisher[i] < trigger[i])
        
        # Fisher extreme levels
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_neutral_long = fisher[i] < -0.5
        fisher_neutral_short = fisher[i] > 0.5
        
        # === CHOPPINESS REGIME ===
        chop_trending = chop_14[i] < 45.0
        chop_ranging = chop_14[i] > 55.0
        chop_neutral = not chop_trending and not chop_ranging
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY LOGIC (REGIME ADAPTIVE + LOOSE FOR TRADES) ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow HTF direction with Fisher confirmation
        if chop_trending:
            if htf_1w_bull and htf_1d_bull:
                # Strong bull: enter on Fisher long cross or oversold
                if fisher_long_cross or fisher_oversold:
                    desired_signal = SIZE_STRONG if fisher_long_cross else SIZE_BASE
            elif htf_1w_bear and htf_1d_bear:
                # Strong bear: enter on Fisher short cross or overbought
                if fisher_short_cross or fisher_overbought:
                    desired_signal = -SIZE_STRONG if fisher_short_cross else -SIZE_BASE
            elif htf_1d_bull:
                # 1d bull only: weaker signal
                if fisher_long_cross or (fisher_oversold and rsi_oversold):
                    desired_signal = SIZE_BASE
            elif htf_1d_bear:
                # 1d bear only: weaker signal
                if fisher_short_cross or (fisher_overbought and rsi_overbought):
                    desired_signal = -SIZE_BASE
        
        # RANGE REGIME: Mean reversion on Fisher extremes
        elif chop_ranging:
            if fisher_oversold and rsi_oversold:
                desired_signal = SIZE_BASE
            elif fisher_overbought and rsi_overbought:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Use Fisher crosses only
        elif chop_neutral:
            if fisher_long_cross:
                desired_signal = SIZE_BASE
            elif fisher_short_cross:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
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
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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
    
    return signals