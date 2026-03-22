#!/usr/bin/env python3
"""
Experiment #576: 1d HMA Trend with Weekly Bias and ATR Stoploss

Hypothesis: After 509 failed strategies, simplicity may be the answer for 1d.
Complex regime filters and multi-indicator ensembles keep failing (Sharpe <= 0).

Why this should work on 1d:
1. 1d timeframe = ~365 bars/year = fewer but higher quality signals
2. HMA(21) provides smooth trend following with less lag than EMA
3. Weekly HMA bias prevents counter-trend entries (major failure mode in 2022)
4. HMA slope confirmation adds momentum filter without being too restrictive
5. Simple logic = MORE TRADES (critical - many strategies had 0 trades)
6. 2.5*ATR stoploss protects against crashes like 2022 (-77%)
7. Position size 0.30 = -23% max on 77% crash (vs -77% with size=1.0)

Key differences from failed #570 (1d Donchian):
- HMA cross instead of Donchian breakout (more frequent signals)
- Weekly HMA bias instead of just daily (stronger trend filter)
- HMA slope confirmation (momentum filter)
- Simpler = more trades = statistical significance

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete (max 0.40 per rules)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_trend_weekly_bias_slope_atr_v1"
timeframe = "1d"
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

def calculate_hma_slope(hma, lookback=5):
    """Calculate HMA slope over lookback period (positive = uptrend)."""
    hma_s = pd.Series(hma)
    slope = hma_s.diff(lookback) / hma_s.shift(lookback)
    return slope.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_21 = calculate_hma(close, 21)
    hma_slope = calculate_hma_slope(hma_21, 5)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(hma_slope[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY HMA TREND BIAS ===
        bull_bias = close[i] > hma_1w_aligned[i]
        bear_bias = close[i] < hma_1w_aligned[i]
        
        # === 1D HMA TREND ===
        price_above_hma = close[i] > hma_21[i]
        price_below_hma = close[i] < hma_21[i]
        
        # === HMA SLOPE CONFIRMATION (momentum) ===
        slope_positive = hma_slope[i] > 0.0
        slope_negative = hma_slope[i] < 0.0
        
        # === ENTRY LOGIC (simple - more trades) ===
        new_signal = 0.0
        
        # Long: Price > 1d HMA + HMA slope up + Weekly bullish bias
        if price_above_hma and slope_positive and bull_bias:
            new_signal = SIZE
        
        # Short: Price < 1d HMA + HMA slope down + Weekly bearish bias
        elif price_below_hma and slope_negative and bear_bias:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if weekly HMA flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias:
                new_signal = 0.0
            if position_side < 0 and bull_bias:
                new_signal = 0.0
        
        # Exit if 1d HMA slope flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and slope_negative:
                new_signal = 0.0
            if position_side < 0 and slope_positive:
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