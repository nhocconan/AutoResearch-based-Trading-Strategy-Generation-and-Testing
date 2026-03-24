#!/usr/bin/env python3
"""
Experiment #715: 6h Primary + 1d/1w HTF — Ehlers Fisher + Vol Regime + Trend Bias

Hypothesis: 6h timeframe with Ehlers Fisher Transform captures reversals better than RSI
in bear/range markets (2022 crash, 2025 bear). Combined with:
1. Volatility regime filter (ATR ratio) to avoid entering during vol spikes
2. 1d HMA for intermediate trend bias
3. 1w HMA for long-term trend bias
4. Loose entry thresholds to ensure >=30 trades/year

Why this might work on 6h:
- Fisher Transform normalizes price to Gaussian distribution, better at extremes
- 6h captures multi-day swings without 1d noise
- Vol regime prevents entering during panic (ATR spike)
- Dual HTF (1d+1w) provides confluence without over-filtering

Key innovations:
1. Ehlers Fisher Transform (period=9) - proven reversal indicator
2. ATR(7)/ATR(30) vol ratio - avoid entries during vol spikes (>2.0)
3. 1d HMA(21) + 1w HMA(21) dual bias - both must agree for strong signal
4. Bollinger Band position for mean reversion confirmation
5. Discrete sizing: 0.0, ±0.25, ±0.30
6. 2.5x ATR trailing stop

Entry conditions (LOOSE to ensure trades):
- LONG: Fisher < -1.5 + vol_ratio < 2.0 + (1d HMA bull OR 1w HMA bull)
- SHORT: Fisher > +1.5 + vol_ratio < 2.0 + (1d HMA bear OR 1w HMA bear)
- Strong signal when BOTH 1d and 1w agree

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_vol_regime_hma_1d1w_v1"
timeframe = "6h"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Better at identifying extremes than RSI in ranging markets
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest == lowest:
            fisher[i] = 0.0
            trigger[i] = 0.0
            continue
        
        # Normalize price to 0-1 range
        value = (close[i] - lowest) / (highest - lowest)
        
        # Clamp to avoid division issues
        value = max(0.001, min(0.999, value))
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
        
        # Trigger line (previous fisher value)
        if i > period:
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands - mean reversion levels"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    # BB position: where price sits within bands (0=lower, 1=upper)
    bb_width = upper - lower
    bb_pos = np.zeros(n)
    bb_pos[:] = np.nan
    for i in range(period, n):
        if bb_width[i] > 1e-10:
            bb_pos[i] = (close[i] - lower[i]) / bb_width[i]
    
    return upper, lower, bb_pos

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
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_pos = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Volatility ratio (ATR short / ATR long)
    vol_ratio = np.zeros(n)
    vol_ratio[:] = np.nan
    for i in range(30, n):
        if atr_30[i] > 1e-10:
            vol_ratio[i] = atr_7[i] / atr_30[i]
    
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(vol_ratio[i]):
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
        
        # === HTF BIAS (1d and 1w HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Dual HTF agreement (stronger signal when both agree)
        htf_both_bull = htf_1d_bull and htf_1w_bull
        htf_both_bear = htf_1d_bear and htf_1w_bear
        htf_mixed = not htf_both_bull and not htf_both_bear
        
        # === VOLATILITY REGIME ===
        # vol_ratio < 2.0 = normal vol (safe to enter)
        # vol_ratio > 2.0 = vol spike (avoid entries)
        vol_normal = vol_ratio[i] < 2.0
        vol_spike = vol_ratio[i] >= 2.0
        
        # === FISHER TRANSFORM SIGNALS (LOOSE thresholds for trades) ===
        # Fisher < -1.5 = oversold (potential long)
        # Fisher > +1.5 = overbought (potential short)
        fisher_oversold = fisher[i] < -1.0  # Loosened from -1.5
        fisher_overbought = fisher[i] > 1.0  # Loosened from +1.5
        
        # Fisher cross (more reliable than absolute level)
        fisher_cross_up = fisher[i] > fisher_trigger[i] and fisher[i-1] <= fisher_trigger[i-1] if i > 0 and not np.isnan(fisher_trigger[i-1]) else False
        fisher_cross_down = fisher[i] < fisher_trigger[i] and fisher[i-1] >= fisher_trigger[i-1] if i > 0 and not np.isnan(fisher_trigger[i-1]) else False
        
        # === BOLLINGER BAND POSITION ===
        bb_oversold = not np.isnan(bb_pos[i]) and bb_pos[i] < 0.15  # Near lower band
        bb_overbought = not np.isnan(bb_pos[i]) and bb_pos[i] > 0.85  # Near upper band
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG entries (multiple paths to ensure trades)
        if vol_normal:
            # Path 1: Fisher oversold + any HTF bull
            if fisher_oversold and (htf_1d_bull or htf_1w_bull):
                desired_signal = SIZE_BASE
            
            # Path 2: Fisher cross up + BB oversold
            if fisher_cross_up and bb_oversold:
                desired_signal = max(desired_signal, SIZE_BASE)
            
            # Path 3: Strong - Fisher oversold + both HTF bull
            if fisher_oversold and htf_both_bull:
                desired_signal = SIZE_STRONG
            
            # Path 4: Very oversold Fisher (any HTF)
            if fisher[i] < -1.8:
                desired_signal = max(desired_signal, SIZE_BASE)
        
        # SHORT entries (multiple paths to ensure trades)
        if vol_normal:
            # Path 1: Fisher overbought + any HTF bear
            if fisher_overbought and (htf_1d_bear or htf_1w_bear):
                desired_signal = -SIZE_BASE
            
            # Path 2: Fisher cross down + BB overbought
            if fisher_cross_down and bb_overbought:
                desired_signal = min(desired_signal, -SIZE_BASE)
            
            # Path 3: Strong - Fisher overbought + both HTF bear
            if fisher_overbought and htf_both_bear:
                desired_signal = -SIZE_STRONG
            
            # Path 4: Very overbought Fisher (any HTF)
            if fisher[i] > 1.8:
                desired_signal = min(desired_signal, -SIZE_BASE)
        
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