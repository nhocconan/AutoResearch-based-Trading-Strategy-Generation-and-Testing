#!/usr/bin/env python3
"""
Experiment #558: 4h Primary + 1d HTF — HMA Trend + RSI Pullback + Choppiness Regime

Hypothesis: 4h timeframe with HMA trend following + RSI pullback entries provides 
optimal balance between trade frequency and quality. Choppiness Index filters regime 
to avoid trend strategies in choppy markets. Simpler than KAMA/ADX approach (#522) 
to ensure adequate trade generation.

Key improvements from failed #522:
1. Simpler HMA instead of KAMA (faster computation, proven track record)
2. Looser RSI entry thresholds (40-60 vs strict extremes) to generate more trades
3. Single HTF (1d) instead of dual (1d+1w) to reduce conflicting signals
4. Clearer regime logic: trend when CHOP<45, range when CHOP>55, transitional otherwise
5. Reduced position tracking complexity - signal-based stoploss only

Strategy logic:
1. 1d HMA(21) = HTF trend bias (call ONCE before loop)
2. 4h HMA(21) = primary trend following
3. 4h HMA(48) = slower trend confirmation
4. 4h RSI(14) = entry timing on pullbacks
5. 4h Choppiness(14) = regime filter (trend vs range)
6. 4h ATR(14) = stoploss distance (2.5x)

Regime-adaptive entries:
- TREND (CHOP<45): Long when price>HMA21>HMA48 + RSI pullback 40-50
- TREND (CHOP<45): Short when price<HMA21<HMA48 + RSI rally 50-60
- RANGE (CHOP>55): Long RSI<35, Short RSI>65 (mean reversion)
- TRANSITION (45-55): Half size, require HTF confirmation

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=10 test
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_chop_regime_1d_v2"
timeframe = "4h"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for HTF trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    hma_21 = calculate_hma(close, period=21)
    hma_48 = calculate_hma(close, period=48)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30
    SIZE_RANGE = 0.25
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
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
            position_side = 0
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(hma_48[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            position_side = 0
            continue
        
        # === HTF BIAS (1d) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND ===
        trend_bull = close[i] > hma_21[i] and hma_21[i] > hma_48[i]
        trend_bear = close[i] < hma_21[i] and hma_21[i] < hma_48[i]
        
        # HMA slope confirmation
        hma21_slope_bull = hma_21[i] > hma_21[i-5] if i >= 5 and not np.isnan(hma_21[i-5]) else False
        hma21_slope_bear = hma_21[i] < hma_21[i-5] if i >= 5 and not np.isnan(hma_21[i-5]) else False
        
        # === CHOPPINESS REGIME ===
        is_trend_regime = chop[i] < 45.0
        is_range_regime = chop[i] > 55.0
        is_transition = not is_trend_regime and not is_range_regime
        
        # === RSI CONDITIONS ===
        rsi_pullback_long = 40.0 <= rsi[i] <= 55.0
        rsi_rally_short = 45.0 <= rsi[i] <= 60.0
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow trend with RSI pullback entries
        if is_trend_regime:
            # Long: HTF bull + 4h trend bull + HMA slope up + RSI pullback
            if htf_bull and trend_bull and hma21_slope_bull and rsi_pullback_long:
                desired_signal = SIZE_TREND
            # Short: HTF bear + 4h trend bear + HMA slope down + RSI rally
            elif htf_bear and trend_bear and hma21_slope_bear and rsi_rally_short:
                desired_signal = -SIZE_TREND
            # Simpler trend entries (ensure trades generate)
            elif htf_bull and trend_bull and rsi[i] < 50.0:
                desired_signal = SIZE_TREND * 0.8
            elif htf_bear and trend_bear and rsi[i] > 50.0:
                desired_signal = -SIZE_TREND * 0.8
        
        # RANGE REGIME: Mean reversion at RSI extremes
        elif is_range_regime:
            if rsi_oversold:
                desired_signal = SIZE_RANGE
            elif rsi_overbought:
                desired_signal = -SIZE_RANGE
            # RSI turning from extreme
            elif rsi[i] < 40.0 and i > 0 and rsi[i] > rsi[i-1]:
                desired_signal = SIZE_RANGE * 0.8
            elif rsi[i] > 60.0 and i > 0 and rsi[i] < rsi[i-1]:
                desired_signal = -SIZE_RANGE * 0.8
        
        # TRANSITION REGIME: Half size, require strong HTF confirmation
        elif is_transition:
            if htf_bull and trend_bull and rsi_oversold:
                desired_signal = SIZE_HALF
            elif htf_bear and trend_bear and rsi_overbought:
                desired_signal = -SIZE_HALF
        
        # === STOPLOSS CHECK (2.5x ATR) ===
        stoploss_triggered = False
        
        if position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            if low[i] < stop_price or (entry_atr > 0 and low[i] < trailing_stop):
                stoploss_triggered = True
            stop_price = max(stop_price, trailing_stop)
        
        if position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            if high[i] > stop_price or (entry_atr > 0 and high[i] > trailing_stop):
                stoploss_triggered = True
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_TREND * 0.9:
            final_signal = SIZE_TREND
        elif desired_signal <= -SIZE_TREND * 0.9:
            final_signal = -SIZE_TREND
        elif desired_signal >= SIZE_RANGE * 0.9:
            final_signal = SIZE_RANGE
        elif desired_signal <= -SIZE_RANGE * 0.9:
            final_signal = -SIZE_RANGE
        elif desired_signal >= SIZE_HALF * 0.9:
            final_signal = SIZE_HALF
        elif desired_signal <= -SIZE_HALF * 0.9:
            final_signal = -SIZE_HALF
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if position_side == 0 or np.sign(final_signal) != position_side:
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
            if position_side != 0:
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals