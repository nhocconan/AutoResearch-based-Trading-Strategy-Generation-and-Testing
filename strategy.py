#!/usr/bin/env python3
"""
Experiment #012: 1d RSI Mean Reversion + Weekly HMA Trend Filter
Hypothesis: Daily timeframe captures major swing reversals. RSI(14) extremes (30/70) with
weekly HMA trend filter provides high-probability entries. Bollinger Band position confirms
oversold/overbought conditions. ATR trailing stop limits drawdown.
Timeframe: 1d (REQUIRED), HTF: 1w via mtf_data helper.
Position sizing: 0.25 base, 0.30 max, discrete levels to minimize fee churn.
Key innovation: Looser RSI thresholds (30/70 vs 20/80) ensure sufficient trade frequency
on daily data while weekly HMA prevents counter-trend trades in strong trends.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_rsi_1w_hma_bb_v1"
timeframe = "1d"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    bandwidth = np.nan_to_num(bandwidth, nan=0.0)
    return upper, lower, bandwidth, sma

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

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

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
    rsi = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Additional trend filters
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels to minimize fee churn (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.30
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_bandwidth[i]) or np.isnan(bb_sma[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias (HTF)
        bull_trend = close[i] > hma_1w_aligned[i]
        bear_trend = close[i] < hma_1w_aligned[i]
        
        # RSI signals (mean reversion) - LOOSER thresholds for more trades
        rsi_oversold = rsi[i] < 35  # Not too extreme, ensures trades
        rsi_overbought = rsi[i] > 65  # Not too extreme, ensures trades
        rsi_extreme_oversold = rsi[i] < 25
        rsi_extreme_overbought = rsi[i] > 75
        
        # Price position vs Bollinger Bands
        price_near_lower = close[i] < bb_lower[i] * 1.01  # Within 1% of lower band
        price_near_upper = close[i] > bb_upper[i] * 0.99  # Within 1% of upper band
        price_below_lower = close[i] < bb_lower[i]  # Below lower band
        price_above_upper = close[i] > bb_upper[i]  # Above upper band
        
        # SMA trend confirmation
        sma_bullish = close[i] > sma_50[i]
        sma_bearish = close[i] < sma_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: RSI oversold + price near/below lower BB + weekly bull trend
        if rsi_oversold and price_near_lower and bull_trend:
            new_signal = SIZE_BASE
        # Secondary: RSI extreme oversold + weekly bull trend (stronger signal)
        elif rsi_extreme_oversold and bull_trend:
            new_signal = SIZE_MAX
        # Tertiary: RSI oversold + price below lower BB (capitulation)
        elif rsi_oversold and price_below_lower:
            new_signal = SIZE_BASE
        # Quaternary: RSI oversold + SMA bullish (trend pullback)
        elif rsi_oversold and sma_bullish and bull_trend:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Primary: RSI overbought + price near/above upper BB + weekly bear trend
        if rsi_overbought and price_near_upper and bear_trend:
            new_signal = -SIZE_BASE
        # Secondary: RSI extreme overbought + weekly bear trend (stronger signal)
        elif rsi_extreme_overbought and bear_trend:
            new_signal = -SIZE_MAX
        # Tertiary: RSI overbought + price above upper BB (euphoria)
        elif rsi_overbought and price_above_upper:
            new_signal = -SIZE_BASE
        # Quaternary: RSI overbought + SMA bearish (trend retrace)
        elif rsi_overbought and sma_bearish and bear_trend:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for daily timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for daily timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals