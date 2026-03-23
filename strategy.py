#!/usr/bin/env python3
"""
Experiment #365: 1h Primary + 4h/1d HTF — Relaxed Regime Strategy for Trade Frequency

Hypothesis: Previous 1h strategies (#355, #358, #360) failed with 0 trades because:
1. Session filters (8-20 UTC) eliminated 60%+ of potential entries
2. Volume filters were too strict (>1.2x avg rarely triggers)
3. CRSI thresholds too narrow for 1h timeframe
4. Too many confluence filters = mutually exclusive conditions

This strategy SIMPLIFIES for 1h:
1. 4h HMA(21) for TREND DIRECTION (not 1d - too slow for 1h entries)
2. 1h RSI(14) with RELAXED thresholds (35/65 instead of 30/70) for entry timing
3. 1h Choppiness Index BINARY regime (>50=range, <50=trend) - no triple regime
4. NO session filter (killing trades)
5. NO volume filter (too strict for 1h)
6. ATR(14) 2.0x trailing stop (tighter for 1h vs 4h)
7. Position size = 0.25 (smaller for lower TF, reduces fee impact)

KEY INSIGHT: 1h needs FEWER filters than 4h/1d, not more. Use HTF for direction,
1h only for entry timing. Target 40-80 trades/year on 1h.

TARGET: 40-80 trades/year, Sharpe > 0.5 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_relaxed_regime_4h_hma_rsi_chop_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
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

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro bias (secondary filter)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% position size for 1h (target 40-80 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION (4h HMA - PRIMARY FILTER) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === MACRO BIAS (1d HMA - SECONDARY FILTER) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index - BINARY) ===
        is_choppy = chop[i] > 50.0  # High choppiness = range regime (mean revert)
        is_trending = chop[i] <= 50.0  # Low choppiness = trend regime (breakout)
        
        # === RELAXED ENTRY CONDITIONS FOR 1h ===
        desired_signal = 0.0
        
        if is_choppy:
            # RANGE REGIME: Mean reversion with relaxed RSI
            # Long: RSI < 40 + price above 4h HMA (bullish bias in range)
            # Short: RSI > 60 + price below 4h HMA (bearish bias in range)
            
            rsi_oversold = rsi_14[i] < 40
            rsi_overbought = rsi_14[i] > 60
            
            if price_above_hma_4h and rsi_oversold:
                # Long oversold in bullish range
                desired_signal = BASE_SIZE
            
            elif price_below_hma_4h and rsi_overbought:
                # Short overbought in bearish range
                desired_signal = -BASE_SIZE
        
        elif is_trending:
            # TREND REGIME: Follow 4h HMA direction
            # Long: price above 4h HMA + RSI > 45 (momentum confirmation)
            # Short: price below 4h HMA + RSI < 55 (momentum confirmation)
            
            rsi_momentum_long = rsi_14[i] > 45
            rsi_momentum_short = rsi_14[i] < 55
            
            if price_above_hma_4h and rsi_momentum_long:
                # Long trend in bullish regime
                desired_signal = BASE_SIZE
            
            elif price_below_hma_4h and rsi_momentum_short:
                # Short trend in bearish regime
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (2.0 * ATR trailing - tighter for 1h) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === RSI EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and rsi_14[i] > 65:
            # Long position: exit when RSI reaches overbought
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 35:
            # Short position: exit when RSI reaches oversold
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Check if trend and regime still valid
            if position_side > 0:
                if price_above_hma_4h:
                    if (is_choppy and rsi_14[i] < 65) or \
                       (is_trending and rsi_14[i] > 45):
                        desired_signal = BASE_SIZE
            elif position_side < 0:
                if price_below_hma_4h:
                    if (is_choppy and rsi_14[i] > 35) or \
                       (is_trending and rsi_14[i] < 55):
                        desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals