#!/usr/bin/env python3
"""
Experiment #620: 6h Primary + 1d/1w HTF — Fisher Transform Reversals + Regime Filter

Hypothesis: 6h timeframe is underexplored (0 experiments). Fisher Transform excels at
catching reversals in bear/range markets (2022 crash, 2025 bear) where RSI fails.
Combined with Choppiness regime filter and 1d/1w HTF bias, this should outperform
RSI-based 6h strategies that failed (#611, #615).

Key differences from failed 6h strategies:
1. Fisher Transform instead of RSI - better reversal detection in bear markets
2. Dual HTF bias (1d + 1w HMA) for trend confirmation
3. Choppiness regime filter to avoid trend-following in chop
4. Simpler entry logic - Fisher extremes + HTF alignment
5. ATR-based stoploss with trailing

Strategy logic:
1. 1w HMA(21) = macro trend bias (very slow)
2. 1d HMA(21) = medium trend bias
3. 6h Fisher(9) = reversal signals (crosses -1.5/+1.5)
4. 6h Choppiness(14) = regime (CHOP>55 = range, CHOP<45 = trend)
5. 6h ATR(14) = volatility + stoploss (2.5*ATR)
6. 6h HMA(21) = local trend confirmation

Regime-adaptive entries:
- TREND (CHOP<45): Follow HTF direction, Fisher confirms momentum
- RANGE (CHOP>55): Mean revert at Fisher extremes (-1.5/+1.5)
- Stoploss: 2.5*ATR trailing from entry

Target: Sharpe>0.40, trades>=30 train (8/year), trades>=3 test
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_chop_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Excellent for catching reversals in bear/range markets
    
    Formula:
    1. Normalize: (close - lowest_low) / (highest_high - lowest_low) - 0.5
    2. Scale: 0.999 * normalized (avoid division by zero)
    3. Fisher: 0.5 * ln((1+value)/(1-value))
    4. Signal: EMA of Fisher
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.nanmax(close[i-period+1:i+1])
        lowest_low = np.nanmin(close[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range < 1e-10:
            fisher[i] = 0.0
        else:
            normalized = (close[i] - lowest_low) / price_range - 0.5
            scaled = 0.999 * normalized
            fisher[i] = 0.5 * np.log((1.0 + scaled) / (1.0 - scaled + 1e-10))
    
    # Signal line is EMA of Fisher
    fisher_series = pd.Series(fisher)
    fisher_signal = fisher_series.ewm(span=3, min_periods=3, adjust=False).mean().values
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR
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

def calculate_rsi(close, period=14):
    """Relative Strength Index - additional filter"""
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
    fisher, fisher_signal = calculate_fisher(close, period=9)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    hma_6h = calculate_hma(close, period=21)
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
        
        if np.isnan(fisher[i]) or np.isnan(chop[i]) or np.isnan(hma_6h[i]):
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
        
        # === 6H LOCAL TREND ===
        local_bull = close[i] > hma_6h[i]
        local_bear = close[i] < hma_6h[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_extreme_oversold = fisher[i] < -2.0
        fisher_extreme_overbought = fisher[i] > 2.0
        
        # Fisher crossing signal line (momentum confirmation)
        fisher_bull_cross = False
        fisher_bear_cross = False
        if i > 0 and not np.isnan(fisher_signal[i-1]):
            fisher_bull_cross = fisher[i] > fisher_signal[i] and fisher[i-1] <= fisher_signal[i-1]
            fisher_bear_cross = fisher[i] < fisher_signal[i] and fisher[i-1] >= fisher_signal[i-1]
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55.0   # Range-bound (mean reversion)
        chop_trend = chop[i] < 45.0   # Trending (trend follow)
        chop_neutral = not chop_range and not chop_trend
        
        # === RSI FILTER (avoid extreme counter-trend) ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === REGIME DETECTION ===
        is_trend_regime = chop_trend
        is_range_regime = chop_range
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow HTF direction with Fisher momentum confirmation
        if is_trend_regime:
            # Long: HTF bull + local bull + Fisher confirming upside
            if htf_bull and local_bull and fisher[i] > -0.5 and fisher[i] > fisher[i-3] if i >= 3 else False:
                desired_signal = SIZE_STRONG
            # Short: HTF bear + local bear + Fisher confirming downside
            elif htf_bear and local_bear and fisher[i] < 0.5 and fisher[i] < fisher[i-3] if i >= 3 else False:
                desired_signal = -SIZE_STRONG
            # Fisher cross confirmation
            elif htf_bull and fisher_bull_cross and rsi[i] > 40:
                desired_signal = SIZE_BASE
            elif htf_bear and fisher_bear_cross and rsi[i] < 60:
                desired_signal = -SIZE_BASE
        
        # RANGE REGIME: Mean revert at Fisher extremes
        elif is_range_regime:
            # Long at extreme oversold Fisher + RSI confirmation
            if fisher_extreme_oversold and rsi_oversold:
                desired_signal = SIZE_BASE
            # Short at extreme overbought Fisher + RSI confirmation
            elif fisher_extreme_overbought and rsi_overbought:
                desired_signal = -SIZE_BASE
            # Fisher recovery from extreme
            elif fisher_oversold and fisher[i] > fisher[i-1] if i > 0 else False:
                desired_signal = SIZE_BASE * 0.8
            elif fisher_overbought and fisher[i] < fisher[i-1] if i > 0 else False:
                desired_signal = -SIZE_BASE * 0.8
        
        # NEUTRAL REGIME: Wait for HTF alignment + Fisher extreme
        else:
            if htf_bull and fisher_extreme_oversold:
                desired_signal = SIZE_BASE * 0.7
            elif htf_bear and fisher_extreme_overbought:
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