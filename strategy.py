#!/usr/bin/env python3
"""
Experiment #607: 6h Primary + 1d/1w HTF — Fisher Transform + Choppiness Regime + Vol Spike Reversion

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). Combining Ehlers Fisher
Transform (excellent for bear market reversals) with Choppiness Index regime detection and
Volatility Spike mean reversion should outperform simple HMA/EMA strategies that failed in
experiments #595, #600, #603.

Key innovations vs failed 6h strategies:
1. Fisher Transform (period=9) - catches reversals at -1.5/+1.5 levels, proven in bear markets
2. Vol Spike detection (ATR(7)/ATR(30) > 2.0) - enters counter-trend after panic
3. Asymmetric regime - only long when 1d HMA bull, only short when 1d HMA bear
4. Choppiness filter - trend follow when CHOP<38.2, mean revert when CHOP>61.8
5. Dual HTF bias - 1w for macro, 1d for medium-term direction

Strategy logic:
1. 1w HMA(21) = macro trend bias (slowest filter)
2. 1d HMA(21) = medium trend bias (entry direction filter)
3. 6h Fisher(9) = entry timing (cross above -1.5 long, cross below +1.5 short)
4. 6h Choppiness(14) = regime (CHOP<38.2 trend, CHOP>61.8 range)
5. 6h ATR ratio(7/30) = vol spike detection (>2.0 = panic, fade the move)
6. 6h RSI(14) = additional confirmation (oversold/overbought extremes)
7. ATR(14)*2.5 stoploss on all positions

Regime-adaptive entries:
- TREND (CHOP<38.2): Fisher cross + HTF alignment + RSI confirmation
- RANGE (CHOP>61.8): Fisher extreme reversal + HTF mean reversion
- VOL SPIKE (ATR ratio>2.0): Counter-trend fade with tight stop

Target: Sharpe>0.40, trades>=120 train (30/year), trades>=15 test
Timeframe: 6h
Size: 0.25 base, 0.30 strong signals
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_chop_volspike_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Excellent for catching reversals in bear markets
    
    Price = 0.67 * ((close - low)/(high - low) - 0.5) + 0.67 * Price_prev
    Fisher = 0.5 * ln((1 + Price)/(1 - Price))
    
    Long signal: Fisher crosses above -1.5
    Short signal: Fisher crosses below +1.5
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    price = np.zeros(n)
    price[:] = np.nan
    
    for i in range(period, n):
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            current_price = 0.67 * ((close[i] - lowest) / price_range - 0.5)
            if i > period:
                current_price += 0.67 * price[i-1]
            
            # Clamp to avoid ln domain errors
            current_price = np.clip(current_price, -0.999, 0.999)
            price[i] = current_price
            fisher[i] = 0.5 * np.log((1 + current_price) / (1 - current_price))
        else:
            if i > period:
                price[i] = price[i-1]
                fisher[i] = fisher[i-1]
    
    return fisher, price

def calculate_atr_ratio(high, low, close, fast_period=7, slow_period=30):
    """
    ATR Ratio for volatility spike detection
    ATR(7) / ATR(30) > 2.0 indicates volatility spike (panic)
    """
    n = len(close)
    if n < slow_period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_fast = pd.Series(tr).ewm(span=fast_period, min_periods=fast_period, adjust=False).mean().values
    atr_slow = pd.Series(tr).ewm(span=slow_period, min_periods=slow_period, adjust=False).mean().values
    
    atr_ratio = np.zeros(n)
    atr_ratio[:] = np.nan
    for i in range(slow_period, n):
        if atr_slow[i] > 1e-10:
            atr_ratio[i] = atr_fast[i] / atr_slow[i]
    
    return atr_ratio

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

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
    Choppiness Index (CHOP) - measures market choppy vs trending
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for medium trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    fisher, fisher_price = calculate_fisher_transform(high, low, close, period=9)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(high, low, close, fast_period=7, slow_period=30)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(atr_ratio[i]):
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
        
        # === HTF BIAS (1w macro + 1d medium) ===
        htf_bull = close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # === CHOPPINESS REGIME ===
        chop_trend = chop[i] < 38.2      # Trending market
        chop_range = chop[i] > 61.8      # Range-bound market
        chop_transition = not chop_trend and not chop_range
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long_cross = False
        fisher_short_cross = False
        fisher_extreme_low = fisher[i] < -1.5
        fisher_extreme_high = fisher[i] > 1.5
        
        if i > 0 and not np.isnan(fisher[i-1]):
            # Fisher crosses above -1.5 (long signal)
            if fisher[i] > -1.5 and fisher[i-1] <= -1.5:
                fisher_long_cross = True
            # Fisher crosses below +1.5 (short signal)
            if fisher[i] < 1.5 and fisher[i-1] >= 1.5:
                fisher_short_cross = True
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 2.0  # Panic condition
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        rsi_extreme_oversold = rsi[i] < 25.0
        rsi_extreme_overbought = rsi[i] > 75.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow HTF direction with Fisher confirmation
        if chop_trend:
            if htf_bull and fisher_long_cross and rsi[i] > 40.0:
                desired_signal = SIZE_STRONG
            elif htf_bear and fisher_short_cross and rsi[i] < 60.0:
                desired_signal = -SIZE_STRONG
            # Fisher recovery in trend direction
            elif htf_bull and fisher_extreme_low and fisher[i] > fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else False:
                desired_signal = SIZE_BASE
            elif htf_bear and fisher_extreme_high and fisher[i] < fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else False:
                desired_signal = -SIZE_BASE
        
        # RANGE REGIME: Mean reversion at Fisher extremes
        elif chop_range:
            if fisher_extreme_low and rsi_oversold:
                desired_signal = SIZE_BASE
            elif fisher_extreme_high and rsi_overbought:
                desired_signal = -SIZE_BASE
            # Fisher reversal from extreme
            elif fisher_extreme_low and fisher[i] > fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else False:
                desired_signal = SIZE_BASE * 0.8
            elif fisher_extreme_high and fisher[i] < fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else False:
                desired_signal = -SIZE_BASE * 0.8
        
        # VOL SPIKE REGIME: Counter-trend fade (mean reversion after panic)
        if vol_spike:
            if rsi_extreme_oversold and close[i] > hma_1w_aligned[i]:
                desired_signal = SIZE_STRONG  # Strong long after panic selloff
            elif rsi_extreme_overbought and close[i] < hma_1w_aligned[i]:
                desired_signal = -SIZE_STRONG  # Strong short after panic rally
            elif rsi_oversold and not htf_bear:
                desired_signal = SIZE_BASE * 0.9
            elif rsi_overbought and not htf_bull:
                desired_signal = -SIZE_BASE * 0.9
        
        # TRANSITION REGIME: Reduced size, require stronger confirmation
        elif chop_transition:
            if htf_bull and fisher_long_cross and rsi_oversold:
                desired_signal = SIZE_BASE * 0.7
            elif htf_bear and fisher_short_cross and rsi_overbought:
                desired_signal = -SIZE_BASE * 0.7
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
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
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
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
    
    return signals