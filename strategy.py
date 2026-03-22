#!/usr/bin/env python3
"""
Experiment #555: 1h MACD Momentum with 4h HMA Trend + BB Regime Filter

Hypothesis: After 500+ failed experiments, the pattern is clear:
1. Too many filters = 0 trades (see #543, #551 with Sharpe=0.000)
2. RSI-based strategies failed badly (#553 Sharpe=-6.923, #548 Sharpe=-1.055)
3. Donchian breakout failed (#554 Sharpe=-3.023)
4. Supertrend strategies all failed

NEW APPROACH for 1h:
1. MACD histogram for momentum (DIFFERENT from failed RSI strategies)
2. 4h HMA for trend bias (proven in multiple strategies)
3. Bollinger Band width for regime (wide=trend, narrow=range)
4. LOOSE entry conditions to ENSURE trades generate
5. Asymmetric position sizing: larger in trend regime, smaller in range

Why MACD instead of RSI:
- RSI mean-reversion failed on crypto trends
- MACD captures momentum continuation better
- Histogram crossing zero is a clean signal
- Different from 490+ failed strategies

Why 1h timeframe:
- 24 bars/day = enough signals without noise
- Not tested much in recent experiments
- Good balance between 15m (too noisy) and 4h (too slow)

Position sizing: 0.25 base, 0.35 in strong trend regime
Stoploss: 2.5 * ATR(14) trailing
Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_macd_momentum_4h_hma_bb_regime_asymmetric_atr_v1"
timeframe = "1h"
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

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma
    return upper.values, lower.values, bandwidth.values, sma.values

def calculate_percentile_rank(series, window=100):
    """Calculate percentile rank of current value over rolling window."""
    s = pd.Series(series)
    def pr(x):
        if len(x) < 2:
            return np.nan
        return (x[:-1] < x[-1]).sum() / (len(x) - 1)
    pr_vals = s.rolling(window=window, min_periods=window).apply(pr, raw=False)
    return pr_vals.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger(close, 20, 2.0)
    
    # Calculate BB bandwidth percentile for regime detection
    bb_bw_pr = calculate_percentile_rank(bb_bandwidth, 100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_TREND = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    # Track MACD histogram for momentum confirmation
    prev_macd_hist = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(macd_hist[i]) or np.isnan(bb_bandwidth[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === BB REGIME DETECTION ===
        # High bandwidth percentile = trend regime
        # Low bandwidth percentile = range regime
        trend_regime = False
        if not np.isnan(bb_bw_pr[i]):
            trend_regime = bb_bw_pr[i] > 0.5  # Top 50% bandwidth = trend
        
        # === MACD MOMENTUM ===
        macd_bullish = macd_hist[i] > 0
        macd_bearish = macd_hist[i] < 0
        
        # MACD histogram turning up (momentum increasing)
        macd_turning_up = macd_hist[i] > prev_macd_hist
        macd_turning_down = macd_hist[i] < prev_macd_hist
        
        # === ENTRY LOGIC (LOOSE to ensure trades) ===
        new_signal = 0.0
        position_size = SIZE_TREND if trend_regime else SIZE_BASE
        
        # Long: 4h bullish + MACD bullish OR MACD turning up
        # Only ONE of these conditions needed (not both)
        if bull_bias and (macd_bullish or macd_turning_up):
            new_signal = position_size
        
        # Short: 4h bearish + MACD bearish OR MACD turning down
        # Only ONE of these conditions needed (not both)
        elif bear_bias and (macd_bearish or macd_turning_down):
            new_signal = -position_size
        
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
        # Exit if 4h HMA flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias:
                new_signal = 0.0
            if position_side < 0 and bull_bias:
                new_signal = 0.0
        
        # === MACD MOMENTUM FADE EXIT ===
        # Exit long if MACD histogram turns negative
        # Exit short if MACD histogram turns positive
        if in_position and new_signal != 0.0:
            if position_side > 0 and macd_bearish and prev_macd_hist >= 0:
                new_signal = 0.0
            if position_side < 0 and macd_bullish and prev_macd_hist <= 0:
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
        prev_macd_hist = macd_hist[i]
    
    return signals