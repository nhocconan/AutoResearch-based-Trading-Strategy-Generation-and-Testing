#!/usr/bin/env python3
"""
Experiment #162: 1d HMA Trend + Weekly Bias + MACD Momentum + BB Regime Filter

Hypothesis: Daily timeframe captures sustained trend moves while weekly HMA provides
stable higher-timeframe bias. MACD histogram momentum confirms entry timing. Bollinger
Band width detects regime (trend vs range) to adjust entry thresholds. This should
work better than pure mean-reversion on 1d (see #150, #156 which failed).

Why 1d might work now:
- Slower timeframe = fewer false signals, lower fee drag
- Weekly HTF filter prevents counter-trend entries during major reversals
- MACD momentum adds timing precision to HMA trend signals
- BB regime filter adapts to market conditions (trend vs range)
- Conservative position sizing (0.25-0.35) limits drawdown during 2022 crash

Learning from failures:
- #150 (1d EMA): Negative Sharpe - too simple, no regime filter
- #156 (1d regime adaptive): Negative Sharpe - over-filtered, too few trades
- Need balance: enough filters to avoid whipsaws, but not so many that trades=0
- Ensure ≥10 trades per symbol by keeping entry thresholds reasonable

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_1w_bias_macd_bb_regime_atr_v1"
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

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / (sma + 1e-10)
    return upper, lower, sma, bandwidth

def calculate_rsi(close, period=14):
    """Calculate RSI using standard formula."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    hma_1d = calculate_hma(close, 21)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    bb_upper, bb_lower, bb_sma, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d[i]) or np.isnan(macd_hist[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_bandwidth[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1w HMA = higher timeframe trend bias (very stable)
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # 1d HMA = primary trend direction
        bull_trend_1d = close[i] > hma_1d[i]
        bear_trend_1d = close[i] < hma_1d[i]
        
        # === REGIME DETECTION via Bollinger Bandwidth ===
        # Low bandwidth = range/compression (expect breakout)
        # High bandwidth = trend/volatile (expect continuation)
        bb_percentile = np.nanpercentile(bb_bandwidth[max(0,i-100):i+1], 50)
        is_range_regime = bb_bandwidth[i] < bb_percentile * 0.8
        is_trend_regime = bb_bandwidth[i] > bb_percentile * 1.2
        
        # === MACD MOMENTUM CONFIRMATION ===
        # MACD histogram crossing above zero = bullish momentum
        # MACD histogram crossing below zero = bearish momentum
        macd_bullish = macd_hist[i] > 0 and macd_hist[i-1] <= 0
        macd_bearish = macd_hist[i] < 0 and macd_hist[i-1] >= 0
        macd_positive = macd_hist[i] > 0
        macd_negative = macd_hist[i] < 0
        
        # === RSI FILTER (not too strict) ===
        # RSI > 45 = not oversold (for longs)
        # RSI < 55 = not overbought (for shorts)
        rsi_ok_long = rsi[i] > 40
        rsi_ok_short = rsi[i] < 60
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Primary: 1d bullish + 1w bullish + MACD positive + RSI ok
        # Secondary: Range regime allows entry at BB lower
        long_condition_1 = bull_trend_1d and bull_trend_1w and macd_positive and rsi_ok_long
        long_condition_2 = bull_trend_1d and macd_bullish and close[i] <= bb_lower[i] * 1.01
        
        if long_condition_1 or long_condition_2:
            # Stronger signal if both 1d and 1w agree
            if bull_trend_1d and bull_trend_1w:
                new_signal = SIZE_STRONG
            else:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Primary: 1d bearish + 1w bearish + MACD negative + RSI ok
        # Secondary: Range regime allows entry at BB upper
        short_condition_1 = bear_trend_1d and bear_trend_1w and macd_negative and rsi_ok_short
        short_condition_2 = bear_trend_1d and macd_bearish and close[i] >= bb_upper[i] * 0.99
        
        if short_condition_1 or short_condition_2:
            # Stronger signal if both 1d and 1w agree
            if bear_trend_1d and bear_trend_1w:
                new_signal = -SIZE_STRONG
            else:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals