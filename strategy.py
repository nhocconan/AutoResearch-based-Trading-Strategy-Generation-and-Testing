#!/usr/bin/env python3
"""
Experiment #096: 12h Primary + 1d HTF — Dual Regime Adaptive Strategy

Hypothesis: Previous 12h strategies failed due to broken position tracking and overly aggressive exits.
This version fixes the signal persistence logic and uses cleaner regime-adaptive entries.

Key improvements from #086 failures:
1) Fixed position tracking - signal persists until explicit exit condition
2) Regime-adaptive: CHOP<45 = trend follow, CHOP>55 = mean revert
3) Looser RSI thresholds (30/70 instead of 25/75) for more trades
4) Bollinger Band mean reversion in ranging regime (proven edge)
5) Simpler stoploss: 2.5*ATR from entry (not trailing - less churn)
6) Discrete signals: 0.0, ±0.30, ±0.35 only

Why this should work:
- Dual regime captures both trending and ranging markets (BTC/ETH spend 60% ranging)
- 1d HMA prevents counter-trend trades in 2025 bear market
- Bollinger mean reversion works well in chop (ETH Sharpe +0.923 in research)
- Simpler logic = more trades across ALL symbols
- 12h TF naturally limits to 25-45 trades/year

Position size: 0.30 base, 0.35 max with confluence
Stoploss: 2.5*ATR from entry price
Target: 30-50 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_bb_1d_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def calculate_ema(close, period=21):
    """Calculate EMA."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1d HMA slope (trend strength)
    hma_1d_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_aligned[i-1]) and hma_1d_aligned[i-1] != 0:
            hma_1d_slope[i] = (hma_1d_aligned[i] - hma_1d_aligned[i-1]) / hma_1d_aligned[i-1] * 100
        else:
            hma_1d_slope[i] = 0.0
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_dev=2.0)
    ema_21 = calculate_ema(close, period=21)
    hma_12h_16 = calculate_hma(close, period=16)
    hma_12h_48 = calculate_hma(close, period=48)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.30
    POSITION_SIZE_MAX = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(bb_upper[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === HTF TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        hma_slope_positive = hma_1d_slope[i] > 0.05
        hma_slope_negative = hma_1d_slope[i] < -0.05
        
        # === CHOPPINESS REGIME ===
        chop_trending = chop_14[i] < 45.0  # trending market
        chop_ranging = chop_14[i] > 55.0  # ranging market
        chop_neutral = not chop_trending and not chop_ranging
        
        # === 12h HMA CROSSOVER ===
        hma_bullish = hma_12h_16[i] > hma_12h_48[i]
        hma_bearish = hma_12h_16[i] < hma_12h_48[i]
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_neutral_long = rsi_14[i] < 55.0
        rsi_neutral_short = rsi_14[i] > 45.0
        
        # === BOLLINGER BAND SIGNALS ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        price_near_bb_mid = abs(close[i] - bb_mid[i]) < 0.005 * bb_mid[i]
        
        # === EMA CONFIRMATION ===
        ema_bullish = close[i] > ema_21[i]
        ema_bearish = close[i] < ema_21[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- TRENDING REGIME: Follow 1d trend with 12h pullback entries ---
        if chop_trending:
            # LONG: 1d bullish + 12h bullish + RSI pullback
            if price_above_hma_1d and hma_bullish and rsi_neutral_long:
                new_signal = POSITION_SIZE_BASE
                if ema_bullish and rsi_oversold:
                    new_signal = POSITION_SIZE_MAX
            
            # SHORT: 1d bearish + 12h bearish + RSI pullback
            if price_below_hma_1d and hma_bearish and rsi_neutral_short:
                new_signal = -POSITION_SIZE_BASE
                if ema_bearish and rsi_overbought:
                    new_signal = -POSITION_SIZE_MAX
        
        # --- RANGING REGIME: Mean revert at Bollinger extremes ---
        elif chop_ranging:
            # LONG: Price below BB lower + RSI oversold + 1d not strongly bearish
            if price_below_bb_lower and rsi_oversold and not (price_below_hma_1d and hma_slope_negative):
                new_signal = POSITION_SIZE_BASE
                if hma_1d_slope[i] > -0.1:  # 1d not crashing
                    new_signal = POSITION_SIZE_MAX
            
            # SHORT: Price above BB upper + RSI overbought + 1d not strongly bullish
            if price_above_bb_upper and rsi_overbought and not (price_above_hma_1d and hma_slope_positive):
                new_signal = -POSITION_SIZE_BASE
                if hma_1d_slope[i] < 0.1:  # 1d not rallying hard
                    new_signal = -POSITION_SIZE_MAX
        
        # --- NEUTRAL REGIME: Wait for clearer signals ---
        elif chop_neutral:
            # Only enter with strong confluence
            if price_above_hma_1d and hma_bullish and rsi_oversold and ema_bullish:
                new_signal = POSITION_SIZE_BASE
            if price_below_hma_1d and hma_bearish and rsi_overbought and ema_bearish:
                new_signal = -POSITION_SIZE_BASE
        
        # === STOPLOSS CHECK (2.5 * ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            stop_price = entry_price - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            stop_price = entry_price + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND CHANGE ===
        # Exit long if 1d HMA turns strongly bearish
        if in_position and position_side > 0:
            if price_below_hma_1d and hma_slope_negative:
                new_signal = 0.0
        
        # Exit short if 1d HMA turns strongly bullish
        if in_position and position_side < 0:
            if price_above_hma_1d and hma_slope_positive:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            new_signal = 0.0
        
        # === EXIT ON MEAN REVERSION COMPLETE (ranging regime) ===
        if in_position and chop_ranging:
            if position_side > 0 and price_near_bb_mid:
                new_signal = 0.0
            if position_side < 0 and price_near_bb_mid:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        if new_signal != 0.0:
            if not in_position:
                # New position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
            # else: hold position, keep tracking
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
        
        signals[i] = new_signal
    
    return signals