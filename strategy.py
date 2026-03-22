#!/usr/bin/env python3
"""
Experiment #478: 4h Adaptive Trend with Daily/Weekly HMA Bias

Hypothesis: After analyzing 477 failed experiments, the key insight is that 4h timeframe
needs SIMPLE but ROBUST logic. Complex regime filters (CHOP, ADX thresholds) caused
too few trades or whipsaws. This strategy uses:

1. DAILY HMA(21) as primary trend filter (via mtf_data helper)
   - Long only when price > 1d HMA
   - Short only when price < 1d HMA
   - HMA is smoother than EMA, less whipsaw

2. WEEKLY HMA(21) as secondary confirmation (via mtf_data helper)
   - Only take longs when 1w HMA also bullish (price > 1w HMA)
   - Shorts allowed regardless of 1w (bear rallies are faster)
   - This asymmetry matches crypto behavior

3. KAMA (Kaufman Adaptive Moving Average) for entry timing
   - KAMA adapts to volatility (fast in trends, slow in chop)
   - Long: price pulls back to KAMA + KAMA sloping up
   - Short: price rallies to KAMA + KAMA sloping down
   - ER (Efficiency Ratio) based adaptation

4. RSI(14) for pullback confirmation (LOOSE thresholds)
   - Long: RSI > 40 (not oversold, just not overbought)
   - Short: RSI < 60 (not overbought, just not oversold)
   - These loose thresholds ensure 20-40 trades/year

5. ATR(14) trailing stop at 2.5x
   - Tighter than daily strategies (4h has less noise)
   - Signal → 0 when price moves 2.5*ATR against position

6. POSITION SIZING: 0.28 discrete
   - Conservative for 4h volatility
   - Discrete levels minimize fee churn

Why this should work on 4h:
- 4h is the "goldilocks" timeframe (not too noisy like 15m, not too slow like 1d)
- Dual HMA filter (1d + 1w) provides robust trend bias
- KAMA adapts to market conditions automatically
- Loose RSI thresholds ensure sufficient trades (>10 per symbol)
- Asymmetric logic (stricter long filter) matches crypto behavior
- Should beat #472 Sharpe of 0.676

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_dual_hma_rsi_asymmetric_atr_v1"
timeframe = "4h"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market volatility via Efficiency Ratio (ER).
    ER = |change| / sum(|change|) over period
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[0:period] = np.nan
    
    noise = np.zeros(n)
    for i in range(period, n):
        noise[i] = np.sum(np.abs(np.diff(close[i-period:i+1])))
    
    er = np.zeros(n)
    for i in range(period, n):
        if noise[i] > 1e-10:
            er[i] = change[i] / noise[i]
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama = calculate_kama(close, 10, 2, 30)
    sma_50 = calculate_sma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
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
        
        if np.isnan(rsi[i]) or np.isnan(kama[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        # === DAILY HMA TREND BIAS ===
        bull_1d = close[i] > hma_1d_aligned[i]
        bear_1d = close[i] < hma_1d_aligned[i]
        
        # === WEEKLY HMA CONFIRMATION ===
        bull_1w = close[i] > hma_1w_aligned[i]
        
        # === KAMA SLOPE ===
        kama_slope_up = kama[i] > kama[i-5] if i >= 5 else False
        kama_slope_down = kama[i] < kama[i-5] if i >= 5 else False
        
        # === ASYMMETRIC ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG: Requires both 1d and 1w bullish (strict filter)
        if bull_1d and bull_1w:
            # Price pullback to KAMA + KAMA sloping up + RSI not overbought
            if kama_slope_up and rsi[i] > 40 and rsi[i] < 70:
                # Price near or above KAMA (pullback entry)
                if close[i] >= kama[i] * 0.995:
                    new_signal = SIZE
        
        # SHORT: Only requires 1d bearish (looser filter for fast bear rallies)
        if bear_1d:
            # Price rally to KAMA + KAMA sloping down + RSI not oversold
            if kama_slope_down and rsi[i] < 60 and rsi[i] > 30:
                # Price near or below KAMA (rally entry)
                if close[i] <= kama[i] * 1.005:
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
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 1d trend flips bearish
        if in_position and position_side > 0 and bear_1d:
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