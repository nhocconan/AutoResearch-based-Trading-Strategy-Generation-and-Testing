#!/usr/bin/env python3
"""
Experiment #357: 1h Regime-Adaptive Strategy with 4h Trend + Bollinger BW + RSI
Hypothesis: 1h timeframe needs regime detection to switch between trend-following and mean-reversion.
Use 4h HMA for macro trend bias, Bollinger Bandwidth percentile for regime (wide=trend, narrow=MR),
RSI(14) for entry timing with loose thresholds (25-75) to ensure sufficient trade frequency.
ATR(14) stoploss at 2.5x protects capital. Position size 0.25-0.35 discrete levels.
Timeframe: 1h (REQUIRED), HTF: 4h for trend via mtf_data helper.
Target: Beat Sharpe=0.499 with 30-80 trades total across train+test.
Key insight: Regime-adaptive logic prevents trend strategies from dying in ranges and vice versa.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_4h_hma_bollinger_rsi_atr_v1"
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
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    return upper, lower, bandwidth

def calculate_bandwidth_percentile(bandwidth, lookback=100):
    """Calculate rolling percentile of bandwidth (regime indicator)."""
    bw_series = pd.Series(bandwidth)
    # Percentile rank: where current BW sits in last lookback bars
    percentile = bw_series.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
    ).values
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    bw_percentile = calculate_bandwidth_percentile(bandwidth, 100)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):  # Start after 150 bars for all indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(bandwidth[i]) or np.isnan(bw_percentile[i]):
            signals[i] = 0.0
            continue
        
        # 4h macro trend bias (SOFT filter - boosts confidence)
        hma_4h_valid = not np.isnan(hma_4h_aligned[i])
        trend_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # Regime detection via Bollinger Bandwidth percentile
        # BW percentile > 0.6 = wide bands = trending regime
        # BW percentile < 0.4 = narrow bands = mean-reversion regime
        trending_regime = bw_percentile[i] > 0.55
        mean_revert_regime = bw_percentile[i] < 0.45
        
        # Price position relative to Bollinger Bands
        price_vs_upper = close[i] >= bb_upper[i]
        price_vs_lower = close[i] <= bb_lower[i]
        price_vs_middle = bb_lower[i] < close[i] < bb_upper[i]
        
        new_signal = 0.0
        
        # === TRENDING REGIME: Follow the 4h trend ===
        if trending_regime:
            # Long: 4h bullish + RSI pullback (not overbought)
            if trend_bullish and rsi[i] < 65 and rsi[i] > 35:
                new_signal = SIZE_ENTRY
            # Short: 4h bearish + RSI pullback (not oversold)
            elif trend_bearish and rsi[i] > 35 and rsi[i] < 65:
                new_signal = -SIZE_ENTRY
            # Momentum breakout: price breaks BB with trend
            elif trend_bullish and price_vs_upper and rsi[i] > 50:
                new_signal = SIZE_ENTRY
            elif trend_bearish and price_vs_lower and rsi[i] < 50:
                new_signal = -SIZE_ENTRY
        
        # === MEAN REVERSION REGIME: Fade extremes ===
        elif mean_revert_regime:
            # Long: Price at lower BB + RSI oversold
            if price_vs_lower and rsi[i] < 35:
                new_signal = SIZE_ENTRY
            # Short: Price at upper BB + RSI overbought
            elif price_vs_upper and rsi[i] > 65:
                new_signal = -SIZE_ENTRY
            # RSI extreme reversal
            elif rsi[i] < 25:
                new_signal = SIZE_ENTRY
            elif rsi[i] > 75:
                new_signal = -SIZE_ENTRY
        
        # === NEUTRAL REGIME: Use RSI extremes only ===
        else:
            if rsi[i] < 30:
                new_signal = SIZE_ENTRY
            elif rsi[i] > 70:
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals