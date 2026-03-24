#!/usr/bin/env python3
"""
Experiment #680: 6h Primary + 1d/1w HTF — Fisher Transform Reversal + Multi-TF Trend Filter

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). Fisher Transform excels 
at catching reversals in bear/range markets (2022 crash, 2025 bear). Combined with 1w/1d HMA 
for major trend bias and Choppiness Index for regime detection. This should outperform pure 
trend strategies that failed in 2022-2024 mixed regimes.

Key innovations:
1. Fisher Transform(9) - normalized oscillator (-1 to +1) for reversal detection
2. 1w HMA(21) - major trend bias (only long above, only short below)
3. 1d HMA(21) - intermediate trend confirmation
4. Choppiness Index(14) - regime filter (>61.8 = range, <38.2 = trend)
5. Asymmetric entries - different logic for range vs trend regimes
6. ATR(14) trailing stop - 2.5x for risk management

Entry conditions (LOOSE to ensure trades):
- RANGE regime (CHOP>61.8): Fisher<-1.5 + price>1w HMA → long, Fisher>+1.5 + price<1w HMA → short
- TREND regime (CHOP<38.2): Fisher cross + 1d HMA alignment + 1w HMA bias
- Size: 0.25 base, 0.30 strong signals

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-50%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_regime_hma_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Excellent for catching reversals in bear/range markets
    """
    n = len(close)
    if n < period + 10:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    # Calculate typical price and normalize
    for i in range(period, n):
        # Highest high and lowest low over period
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        
        range_val = highest - lowest
        if range_val < 1e-10:
            continue
        
        # Normalize price to 0-1 range
        price_norm = (close[i] - lowest) / range_val
        
        # Clamp to avoid extreme values
        price_norm = max(0.001, min(0.999, price_norm))
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + price_norm) / (1 - price_norm))
        
        # Trigger line (1-period lag of fisher)
        if i > period:
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    >61.8 = choppy/range, <38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            continue
        
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        if atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_hma(close, period):
    """Hull Moving Average - smoother and more responsive than EMA"""
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

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.maximum(delta, 0)
    loss[1:] = np.maximum(-delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    rs[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
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
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, period=9)
    chop = calculate_choppiness_index(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
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
        
        if np.isnan(fisher[i]) or np.isnan(chop[i]):
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
        htf_weekly_bull = close[i] > hma_1w_aligned[i]
        htf_weekly_bear = close[i] < hma_1w_aligned[i]
        htf_daily_bull = close[i] > hma_1d_aligned[i]
        htf_daily_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop[i] > 61.8
        is_trend = chop[i] < 38.2
        # Neutral zone: 38.2 to 61.8
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher[i] > fisher_trigger[i] and fisher[i-1] <= fisher_trigger[i-1] if i > 0 and not np.isnan(fisher_trigger[i-1]) else False
        fisher_cross_down = fisher[i] < fisher_trigger[i] and fisher[i-1] >= fisher_trigger[i-1] if i > 0 and not np.isnan(fisher_trigger[i-1]) else False
        
        # === RSI FILTER ===
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADES) ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion at extremes
        if is_range:
            # Long: weekly bull + fisher oversold + RSI oversold
            if htf_weekly_bull and fisher_oversold and rsi_oversold:
                desired_signal = SIZE_STRONG
            # Short: weekly bear + fisher overbought + RSI overbought
            elif htf_weekly_bear and fisher_overbought and rsi_overbought:
                desired_signal = -SIZE_STRONG
            # Weaker range entries
            elif htf_weekly_bull and fisher_oversold:
                desired_signal = SIZE_BASE
            elif htf_weekly_bear and fisher_overbought:
                desired_signal = -SIZE_BASE
        
        # TREND REGIME: Follow direction with Fisher pullback entries
        elif is_trend:
            # Long: weekly bull + daily bull + fisher cross up from oversold
            if htf_weekly_bull and htf_daily_bull and fisher_cross_up:
                desired_signal = SIZE_STRONG
            # Short: weekly bear + daily bear + fisher cross down from overbought
            elif htf_weekly_bear and htf_daily_bear and fisher_cross_down:
                desired_signal = -SIZE_STRONG
            # Weaker trend entries
            elif htf_weekly_bull and htf_daily_bull and fisher[i] < 0:
                desired_signal = SIZE_BASE
            elif htf_weekly_bear and htf_daily_bear and fisher[i] > 0:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Conservative, require stronger signals
        else:
            # Long: all HTF bull + fisher cross up
            if htf_weekly_bull and htf_daily_bull and fisher_cross_up and rsi_oversold:
                desired_signal = SIZE_BASE
            # Short: all HTF bear + fisher cross down
            elif htf_weekly_bear and htf_daily_bear and fisher_cross_down and rsi_overbought:
                desired_signal = -SIZE_BASE
        
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
        elif abs(desired_signal) >= SIZE_BASE * 0.4:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.5
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