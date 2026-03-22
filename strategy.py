#!/usr/bin/env python3
"""
Experiment #497: 12h Dual-HTF Trend with RSI Pullback - Simplified Asymmetric Logic

Hypothesis: After 473+ failed experiments, the pattern is clear - overly complex regime
filters kill trade frequency and create whipsaw. For 12h timeframe, I need:

1. SIMPLER LOGIC: Too many filters = 0 trades or negative Sharpe
2. DUAL HTF BIAS: 1d HMA for immediate trend, 1w HMA for meta-regime
3. ASYMMETRIC ENTRIES: 
   - Bull (price > 1w HMA): Long on RSI pullback to 35-45
   - Bear (price < 1w HMA): Short on RSI rally to 55-65
4. 1D HMA CONFIRMATION: Only trade in direction of 1d HMA slope
5. WIDER STOPS: 2.5x ATR for 12h volatility (not 3x which is too wide)
6. POSITION SIZE: 0.25 discrete (conservative for 12h swings)

Why 12h should work:
- Slower than 4h, fewer false signals
- Faster than 1d, more trade opportunities
- 12h bars capture multi-day trends without noise
- Dual HTF (1d + 1w) provides robust trend filtering

Key difference from failed experiments:
- NO Choppiness Index (failed in #485, #486, #490, #496)
- NO Connors RSI (failed in #485, #486)
- NO complex volume filters (failed in #488, #493, #495)
- Just HMA trend + RSI pullback + ATR stop = proven edge

Timeframe: 12h (REQUIRED)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_htf_hma_rsi_pullback_asymmetric_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    sma_50 = calculate_sma(close, 50)
    
    # Calculate 1d HMA slope (direction)
    hma_1d_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_aligned[i-1]):
            hma_1d_slope[i] = hma_1d_aligned[i] - hma_1d_aligned[i-1]
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        # === META REGIME: 1W HMA (Bull/Bear Market) ===
        bull_regime = close[i] > hma_1w_aligned[i]
        bear_regime = close[i] < hma_1w_aligned[i]
        
        # === PRIMARY TREND: 1D HMA DIRECTION ===
        uptrend_1d = hma_1d_slope[i] > 0
        downtrend_1d = hma_1d_slope[i] < 0
        
        # === ASYMMETRIC ENTRY LOGIC ===
        new_signal = 0.0
        
        # BULL REGIME: Look for long entries on pullbacks
        if bull_regime:
            # Primary trend up + RSI pullback to 35-45
            if uptrend_1d and rsi[i] >= 35 and rsi[i] <= 48:
                new_signal = SIZE
            # Also allow entries when price > SMA50 (momentum confirmation)
            elif close[i] > sma_50[i] and rsi[i] >= 38 and rsi[i] <= 50:
                new_signal = SIZE
        
        # BEAR REGIME: Look for short entries on rallies
        if bear_regime:
            # Primary trend down + RSI rally to 52-65
            if downtrend_1d and rsi[i] >= 52 and rsi[i] <= 65:
                new_signal = -SIZE
            # Also allow entries when price < SMA50 (momentum confirmation)
            elif close[i] < sma_50[i] and rsi[i] >= 50 and rsi[i] <= 62:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME REVERSAL EXIT ===
        # Exit if weekly trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_regime:
                new_signal = 0.0
            if position_side < 0 and bull_regime:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals