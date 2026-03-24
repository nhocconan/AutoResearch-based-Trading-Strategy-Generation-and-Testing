#!/usr/bin/env python3
"""
Experiment #752: 12h Primary + 1d HTF — Asymmetric Regime with Volatility Filter

Hypothesis: 12h timeframe captures multi-day swings while avoiding noise. 
1d HTF provides reliable trend bias. Key innovation: ASYMMETRIC regime logic
that adapts to market conditions (trend vs range) using ADX + ATR ratio.

Why this should work:
1. 12h is proven timeframe (20-50 trades/year optimal)
2. 1d HMA(21) gives clean trend bias without whipsaw
3. ADX(14) distinguishes trending (ADX>25) vs ranging (ADX<20) markets
4. ATR ratio (ATR7/ATR30) detects vol spikes for mean-reversion entries
5. Asymmetric sizing: larger positions in trending regimes, smaller in chop
6. Loose RSI thresholds (40/60) ensure trade generation across all symbols

Entry Logic:
- TREND REGIME (ADX>25): Follow 1d bias, enter on 12h RSI pullback (40/60)
- RANGE REGIME (ADX<20): Mean revert at extremes (RSI<35 long, RSI>65 short)
- TRANSITION (ADX 20-25): Half size, require HMA crossover confirmation

Risk Management:
- ATR(14) 2.5x trailing stop on all positions
- Discrete sizing: 0.0, ±0.20, ±0.25, ±0.30
- Position tracking with stoploss → signal=0 when hit

Target: Sharpe>0.45, trades≥30 train, trades≥3 test per symbol, DD>-35%
Timeframe: 12h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_asymmetric_regime_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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
    """Relative Strength Index - momentum oscillator"""
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

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # ATR ratio for vol spike detection
    atr_ratio = np.zeros(n)
    atr_ratio[:] = np.nan
    for i in range(30, n):
        if atr_30[i] > 1e-10:
            atr_ratio[i] = atr_7[i] / atr_30[i]
    
    signals = np.zeros(n)
    
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(adx_14[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION ===
        adx = adx_14[i]
        vol_ratio = atr_ratio[i]
        
        # Trend regime: ADX > 25
        # Range regime: ADX < 20
        # Transition: ADX 20-25
        is_trend = adx > 25.0
        is_range = adx < 20.0
        is_transition = not is_trend and not is_range
        
        # Vol spike: ATR7/ATR30 > 2.0
        vol_spike = vol_ratio > 2.0
        
        # === 12h HMA CROSSOVER ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_16[i-1]) and not np.isnan(hma_48[i-1]):
            hma_cross_long = (hma_16[i-1] <= hma_48[i-1]) and (hma_16[i] > hma_48[i])
            hma_cross_short = (hma_16[i-1] >= hma_48[i-1]) and (hma_16[i] < hma_48[i])
        
        # === 12h HMA TREND ===
        hma_bull = hma_16[i] > hma_48[i]
        hma_bear = hma_16[i] < hma_48[i]
        
        # === RSI CONDITIONS ===
        rsi = rsi_14[i]
        rsi_oversold = rsi < 40.0
        rsi_overbought = rsi > 60.0
        rsi_extreme_low = rsi < 35.0
        rsi_extreme_high = rsi > 65.0
        
        # === ENTRY LOGIC (ASYMMETRIC BY REGIME) ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        if is_trend:
            # TREND REGIME: Follow HTF bias, enter on pullback
            if htf_bull and (rsi_oversold or hma_cross_long):
                if htf_bull and hma_bull:
                    signal_strength = 0.30  # Strong trend alignment
                else:
                    signal_strength = 0.25  # HTF bias only
                desired_signal = signal_strength
            
            elif htf_bear and (rsi_overbought or hma_cross_short):
                if htf_bear and hma_bear:
                    signal_strength = -0.30
                else:
                    signal_strength = -0.25
                desired_signal = signal_strength
        
        elif is_range:
            # RANGE REGIME: Mean reversion at extremes
            if rsi_extreme_low and htf_bull:
                desired_signal = 0.25
            elif rsi_extreme_low and vol_spike:
                # Vol spike + oversold = strong mean reversion signal
                desired_signal = 0.30
            elif rsi_extreme_high and htf_bear:
                desired_signal = -0.25
            elif rsi_extreme_high and vol_spike:
                desired_signal = -0.30
        
        else:  # is_transition
            # TRANSITION: Require HMA crossover confirmation, half size
            if htf_bull and hma_cross_long:
                desired_signal = 0.20
            elif htf_bear and hma_cross_short:
                desired_signal = -0.20
        
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
        if desired_signal >= 0.28:
            final_signal = 0.30
        elif desired_signal >= 0.23:
            final_signal = 0.25
        elif desired_signal >= 0.18:
            final_signal = 0.20
        elif desired_signal <= -0.28:
            final_signal = -0.30
        elif desired_signal <= -0.23:
            final_signal = -0.25
        elif desired_signal <= -0.18:
            final_signal = -0.20
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