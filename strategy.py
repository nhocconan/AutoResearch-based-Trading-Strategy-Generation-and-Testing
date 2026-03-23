#!/usr/bin/env python3
"""
Experiment #056: 12h Primary + 1d HTF — KAMA Adaptive Trend + Fisher Transform + Choppiness Regime

Hypothesis: 12h timeframe with 1d trend bias using KAMA (adaptive to market efficiency) + 
Fisher Transform (reversal detection) + Choppiness regime filter will generate 25-50 trades/year
with Sharpe > 0.486 (beat current best).

Key insights from 55 failed experiments:
1) 12h timeframe proven to work (exp #049 kept with Sharpe=-0.229, exp #052 0 trades)
2) KAMA adapts to market efficiency - less whipsaw than EMA/HMA in choppy markets
3) Fisher Transform catches reversals better than RSI in bear/range markets (exp #047 kept)
4) Choppiness regime filter switches between trend/mean-revert modes automatically
5) Too many filters = 0 trades (exp #052, #055 had Sharpe=0.000)
6) Simple entry logic with HTF bias works better than complex confluence

Why this should work:
- 12h primary = fewer trades, less fee drag (target 25-50/year)
- 1d HTF KAMA slope = macro trend bias without over-filtering
- KAMA(21) = adaptive trend following (ER adjusts smoothing)
- Fisher(9) = reversal detection at extremes (better than RSI in bears)
- Choppiness(14) = regime detection for adaptive entry logic
- ATR(14) trailing stop = risk management

Position size: 0.28 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
Target: 25-50 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_fisher_chop_regime_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)."""
    close_s = pd.Series(close)
    change = np.abs(close_s.diff())
    volatility = pd.Series(change).rolling(window=period, min_periods=period).sum()
    direction = np.abs(close_s.diff(period))
    er = direction / (volatility + 1e-10)
    er = er.fillna(0.5)
    
    sc = (er * (2.0/(fast+1) - 2.0/(slow+1)) + 2.0/(slow+1)) ** 2
    kama = pd.Series(index=close_s.index, dtype=float)
    kama.iloc[period-1] = close_s.iloc[period-1]
    
    for i in range(period, len(close_s)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama.iloc[i-1])
    
    return kama.values

def calculate_fisher(close, period=9):
    """Calculate Ehlers Fisher Transform."""
    close_s = pd.Series(close)
    highest = close_s.rolling(window=period, min_periods=period).max()
    lowest = close_s.rolling(window=period, min_periods=period).min()
    price_range = highest - lowest + 1e-10
    normalized = (close_s - lowest) / price_range * 2.0 - 1.0
    normalized = normalized.clip(-0.999, 0.999)
    fisher = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
    fisher_prev = fisher.shift(1)
    fisher = fisher.fillna(0.0)
    fisher_prev = fisher_prev.fillna(0.0)
    return fisher.values, fisher_prev.values

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = period
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d KAMA for macro trend bias
    kama_1d = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_21 = calculate_kama(close, period=21)
    fisher, fisher_prev = calculate_fisher(close, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    sma_50 = calculate_sma(close, period=50)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28  # Discrete, within 0.20-0.35 range
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):  # Warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(kama_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]) or np.isnan(kama_21[i]):
            continue
        if np.isnan(sma_50[i]) or atr_14[i] == 0:
            continue
        
        # === 1D MACRO TREND BIAS ===
        kama_1d_slope_up = kama_1d_aligned[i] > kama_1d_aligned[i-1] if i > 0 else False
        kama_1d_slope_down = kama_1d_aligned[i] < kama_1d_aligned[i-1] if i > 0 else False
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # === 12H TREND CONFIRMATION ===
        kama_21_slope_up = kama_21[i] > kama_21[i-3] if i > 3 else False
        kama_21_slope_down = kama_21[i] < kama_21[i-3] if i > 3 else False
        price_above_kama_12h = close[i] > kama_21[i]
        price_below_kama_12h = close[i] < kama_21[i]
        price_above_sma_50 = close[i] > sma_50[i]
        price_below_sma_50 = close[i] < sma_50[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 58.0  # Range market
        is_trending = chop_value < 42.0  # Trend market (with hysteresis)
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = fisher[i] > -1.5 and fisher_prev[i] <= -1.5
        fisher_cross_down = fisher[i] < 1.5 and fisher_prev[i] >= 1.5
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- TRENDING REGIME: KAMA Trend + Fisher Confirmation ---
        if is_trending:
            # Long: Price above KAMA + Fisher cross up + 1d bias
            if price_above_kama_12h and kama_21_slope_up:
                if fisher_cross_up or fisher_oversold:
                    if price_above_kama_1d or kama_1d_slope_up:
                        new_signal = POSITION_SIZE
            
            # Short: Price below KAMA + Fisher cross down + 1d bias
            elif price_below_kama_12h and kama_21_slope_down:
                if fisher_cross_down or fisher_overbought:
                    if price_below_kama_1d or kama_1d_slope_down:
                        new_signal = -POSITION_SIZE
        
        # --- RANGING REGIME: Fisher Mean Reversion ---
        elif is_ranging:
            # Long: Fisher oversold + price above SMA50 (bullish bias)
            if fisher_oversold and price_above_sma_50:
                new_signal = POSITION_SIZE
            
            # Short: Fisher overbought + price below SMA50 (bearish bias)
            elif fisher_overbought and price_below_sma_50:
                new_signal = -POSITION_SIZE
        
        # --- NEUTRAL REGIME: Simple breakout (ensures trades happen) ---
        else:
            # Long: Price above KAMA + 1d bias
            if price_above_kama_12h and price_above_kama_1d:
                new_signal = POSITION_SIZE
            # Short: Price below KAMA + 1d bias
            elif price_below_kama_12h and price_below_kama_1d:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC (avoid premature exits) ===
        if in_position and new_signal == 0.0:
            # Hold long if Fisher not overbought
            if position_side > 0 and fisher[i] < 1.5:
                new_signal = signals[i-1] if i > 0 else 0.0
            # Hold short if Fisher not oversold
            elif position_side < 0 and fisher[i] > -1.5:
                new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND CHANGE ===
        if in_position and position_side > 0:
            if price_below_kama_12h and price_below_kama_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_kama_12h and price_above_kama_1d:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals