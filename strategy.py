#!/usr/bin/env python3
"""
Experiment #674: 1d Primary + 1w HTF — Dual Regime Strategy (Trend + Mean Reversion)

Hypothesis: Daily timeframe with regime detection via Choppiness Index can adapt 
to both trending and ranging markets. Use 1w HMA for long-term bias, then:
- When CHOP < 45 (trending): Follow HMA crossover direction with RSI pullback entries
- When CHOP > 55 (ranging): Mean revert at Bollinger Band extremes
- When 45 <= CHOP <= 55 (transition): Reduce size or stay flat

Key innovations:
1. Choppiness Index(14) regime detection - proven meta-filter for bear/range markets
2. 1w HMA(21) bias filter - only long when above weekly trend, only short when below
3. HMA(16/48) crossover on 1d - faster trend detection than EMA
4. RSI(14) pullback entries - enter on dips in trends, not breakouts
5. Bollinger(20,2.0) for range extremes - fade when CHOP high
6. ATR(14) trailing stop - 2.5x for risk management
7. Discrete sizing: 0.0, ±0.20, ±0.30 to minimize fee churn

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 1d
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_hma_rsi_1w_v1"
timeframe = "1d"
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
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0.0)
    loss[1:] = np.where(delta < 0, -delta, 0.0)
    
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending
    Values > 61.8 = choppy/ranging, Values < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.nanmax(high[i - period + 1:i + 1])
        lowest_low = np.nanmin(low[i - period + 1:i + 1])
        
        if highest_high - lowest_low < 1e-10:
            chop[i] = 100.0
            continue
        
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100.0 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    hma_fast = calculate_hma(close, period=16)
    hma_slow = calculate_hma(close, period=48)
    rsi = calculate_rsi(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_mult=2.0)
    atr = calculate_atr(high, low, close, period=14)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === HMA CROSSOVER (1d) ===
        hma_bull = hma_fast[i] > hma_slow[i]
        hma_bear = hma_fast[i] < hma_slow[i]
        
        # === HMA SLOPE ===
        hma_slope_bull = False
        hma_slope_bear = False
        if i >= 2 and not np.isnan(hma_fast[i-2]):
            hma_slope_bull = hma_fast[i] > hma_fast[i-1] > hma_fast[i-2]
            hma_slope_bear = hma_fast[i] < hma_fast[i-1] < hma_fast[i-2]
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi[i] < 45.0
        rsi_overbought = rsi[i] > 55.0
        rsi_extreme_low = rsi[i] < 35.0
        rsi_extreme_high = rsi[i] > 65.0
        
        # === CHOPPINESS REGIME ===
        regime_trending = chop[i] < 45.0
        regime_ranging = chop[i] > 55.0
        regime_transition = not regime_trending and not regime_ranging
        
        # === BOLLINGER POSITION ===
        near_bb_lower = close[i] <= bb_lower[i] * 1.002
        near_bb_upper = close[i] >= bb_upper[i] * 0.998
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADES) ===
        desired_signal = 0.0
        
        # TRENDING REGIME: Follow HMA crossover with RSI pullback
        if regime_trending:
            # LONG: HTF bull + HMA bull + RSI pullback (not overbought)
            if htf_bull and hma_bull and rsi_oversold:
                desired_signal = SIZE_STRONG
            elif htf_bull and hma_bull and hma_slope_bull:
                desired_signal = SIZE_BASE
            
            # SHORT: HTF bear + HMA bear + RSI pullback (not oversold)
            elif htf_bear and hma_bear and rsi_overbought:
                desired_signal = -SIZE_STRONG
            elif htf_bear and hma_bear and hma_slope_bear:
                desired_signal = -SIZE_BASE
        
        # RANGING REGIME: Mean revert at Bollinger extremes
        elif regime_ranging:
            # LONG: Price at BB lower + RSI oversold + HTF not strongly bear
            if near_bb_lower and rsi_extreme_low:
                desired_signal = SIZE_BASE
            elif near_bb_lower and rsi_oversold and not htf_bear:
                desired_signal = SIZE_BASE * 0.5
            
            # SHORT: Price at BB upper + RSI overbought + HTF not strongly bull
            elif near_bb_upper and rsi_extreme_high:
                desired_signal = -SIZE_BASE
            elif near_bb_upper and rsi_overbought and not htf_bull:
                desired_signal = -SIZE_BASE * 0.5
        
        # TRANSITION REGIME: Smaller positions, wait for confirmation
        elif regime_transition:
            # Only enter on strong confluence
            if htf_bull and hma_bull and hma_slope_bull and rsi_oversold:
                desired_signal = SIZE_BASE * 0.5
            elif htf_bear and hma_bear and hma_slope_bear and rsi_overbought:
                desired_signal = -SIZE_BASE * 0.5
        
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