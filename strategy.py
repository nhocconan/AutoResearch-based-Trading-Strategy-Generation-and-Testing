#!/usr/bin/env python3
"""
Experiment #564: 1d Regime-Adaptive Asymmetric Strategy with Weekly HMA

Hypothesis: After 500+ failed experiments, the key insight is:
1. 1d timeframe needs LOOSE entry conditions to generate enough trades
2. Regime detection (choppy vs trending) determines which logic to use
3. Asymmetric entries: easier to long in bull, easier to short in bear
4. Weekly HMA provides macro trend bias without being too restrictive
5. RSI extremes + BB regime = mean reversion in chop, breakout in trend
6. Conservative sizing (0.25-0.35) protects against 2022-style crashes

Why this should work on 1d:
- 1d has ~365 bars/year = need ~3-5 trades/year minimum per symbol
- Regime-adaptive logic works in both bull and bear markets
- Weekly HMA via mtf_data helper ensures proper alignment
- RSI(14) extremes (25/75) trigger more often than extreme (10/90)
- BB Width percentile detects chop vs trend reliably
- 2.5*ATR stoploss protects capital

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_adaptive_asymmetric_weekly_hma_rsi_bb_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bb(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper.values, lower.values, sma.values

def calculate_bb_width(upper, lower, middle):
    """Calculate Bollinger Band Width (normalized)."""
    width = (upper - lower) / middle
    return width

def calculate_percentile_rank(series, window=100):
    """Calculate percentile rank of current value over rolling window."""
    result = pd.Series(series).rolling(window=window, min_periods=window).apply(
        lambda x: (x[-1] > x[:-1]).sum() / len(x[:-1]) if len(x) > 1 else 0.5
    )
    return result.values

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
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_middle = calculate_bb(close, 20, 2.0)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_middle)
    bb_width_pct = calculate_percentile_rank(bb_width, 100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.35
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        # BB Width percentile: <30 = trending, >70 = choppy
        is_trending = bb_width_pct[i] < 0.30
        is_choppy = bb_width_pct[i] > 0.70
        
        # === WEEKLY HMA TREND BIAS (ASYMMETRIC) ===
        # Bull: price > HMA + 1% buffer (stricter for longs)
        # Bear: price < HMA - 1% buffer (stricter for shorts)
        hma_buffer = hma_1w_aligned[i] * 0.01
        bull_bias = close[i] > hma_1w_aligned[i] + hma_buffer
        bear_bias = close[i] < hma_1w_aligned[i] - hma_buffer
        neutral = not bull_bias and not bear_bias
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE + ASYMMETRIC) ===
        new_signal = 0.0
        
        if is_trending:
            # Trend following: enter on RSI pullback in direction of trend
            # Long: bull bias + RSI 40-55 (pullback, not oversold)
            if bull_bias and 40 <= rsi_14[i] <= 55:
                new_signal = SIZE_BASE
            
            # Short: bear bias + RSI 45-60 (pullback, not overbought)
            elif bear_bias and 45 <= rsi_14[i] <= 60:
                new_signal = -SIZE_BASE
        
        elif is_choppy:
            # Mean reversion: enter at extremes against recent move
            # Long: RSI < 30 (oversold) + price near lower BB
            if rsi_14[i] < 30 and close[i] <= bb_lower[i] * 1.01:
                new_signal = SIZE_BASE
            
            # Short: RSI > 70 (overbought) + price near upper BB
            elif rsi_14[i] > 70 and close[i] >= bb_upper[i] * 0.99:
                new_signal = -SIZE_BASE
        
        else:
            # Neutral regime: only take strong signals
            # Long: RSI < 25 (very oversold)
            if rsi_14[i] < 25:
                new_signal = SIZE_BASE * 0.8
            
            # Short: RSI > 75 (very overbought)
            elif rsi_14[i] > 75:
                new_signal = -SIZE_BASE * 0.8
        
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
        # Exit if weekly HMA flips strongly against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias:
                new_signal = 0.0
            if position_side < 0 and bull_bias:
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