#!/usr/bin/env python3
"""
Experiment #195: 1h Primary + 4h/1d HTF — Simplified Regime + RSI + Session Filter

Hypothesis: Previous 1h strategies failed (0 trades) because filters were too strict.
This version uses:
1. Simpler regime detection (CHOP only, not multiple indicators)
2. Looser RSI thresholds (30/70 instead of 15/85) to generate more trades
3. 4h HMA for trend direction (not 12h+1d which is too restrictive)
4. Session filter (8-20 UTC) to reduce low-liquidity trades
5. Position sizing: 0.25-0.30 (conservative for 1h timeframe)

Key difference from failed #185, #190:
- Fewer confluence requirements (HTF trend OR neutral, not AND)
- Looser RSI extremes (30/70 vs 20/80)
- No volume spike requirement (causes 0 trades on quiet days)
- Session filter reduces trades naturally without killing signal generation

TARGET: 40-70 trades/year, Sharpe > 0.3 on ALL symbols
Position sizing: 0.0, ±0.25, ±0.30 (discrete to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_rsi_session_4h1d_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    hull = (2.0 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hull.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        range_val = highest - lowest
        if range_val < 1e-10 or tr_sum < 1e-10:
            chop[i] = 50.0
        else:
            chop[i] = 100.0 * np.log10(tr_sum / range_val) / np.log10(period)
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Calculate 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === HTF MACRO BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # Bias determination (looser than failed strategies)
        bullish_bias = price_above_hma_4h
        bearish_bias = price_below_hma_4h
        
        # === REGIME DETECTION ===
        is_range = chop_14[i] > 55.0
        is_trend = chop_14[i] < 45.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Only trade during session hours (reduces trades naturally)
        if in_session:
            if is_range:
                # MEAN REVERSION MODE (RSI extremes + Bollinger)
                # Long: RSI < 35 + price near BB lower + bullish or neutral bias
                if rsi_14[i] < 35 and close[i] <= bb_lower[i] * 1.002:
                    if not bearish_bias:  # Allow long if not strongly bearish
                        new_signal = POSITION_SIZE_FULL
                    elif price_above_hma_1d:  # Or if daily is bullish
                        new_signal = POSITION_SIZE_HALF
                
                # Short: RSI > 65 + price near BB upper + bearish or neutral bias
                elif rsi_14[i] > 65 and close[i] >= bb_upper[i] * 0.998:
                    if not bullish_bias:  # Allow short if not strongly bullish
                        new_signal = -POSITION_SIZE_FULL
                    elif price_below_hma_1d:  # Or if daily is bearish
                        new_signal = -POSITION_SIZE_HALF
            
            elif is_trend:
                # TREND FOLLOWING MODE (RSI pullback in direction of trend)
                # Long: RSI 40-55 (pullback) + price above 4h HMA + bullish bias
                if 40 <= rsi_14[i] <= 55 and price_above_hma_4h:
                    if bullish_bias:
                        new_signal = POSITION_SIZE_FULL
                    elif price_above_hma_1d:
                        new_signal = POSITION_SIZE_HALF
                
                # Short: RSI 45-60 (pullback) + price below 4h HMA + bearish bias
                elif 45 <= rsi_14[i] <= 60 and price_below_hma_4h:
                    if bearish_bias:
                        new_signal = -POSITION_SIZE_FULL
                    elif price_below_hma_1d:
                        new_signal = -POSITION_SIZE_HALF
            
            else:
                # NEUTRAL REGIME (45-55 CHOP) - use simple RSI extremes
                # Long: RSI < 30
                if rsi_14[i] < 30 and not bearish_bias:
                    new_signal = POSITION_SIZE_HALF
                
                # Short: RSI > 70
                elif rsi_14[i] > 70 and not bullish_bias:
                    new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and trend still valid (more lenient)
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price still above 4h HMA or RSI not overbought
                if price_above_hma_4h or rsi_14[i] < 70:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price still below 4h HMA or RSI not oversold
                if price_below_hma_4h or rsi_14[i] > 30:
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
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below 4h HMA significantly (2% buffer)
        if in_position and position_side > 0 and close[i] < hma_4h_aligned[i] * 0.98:
            new_signal = 0.0
        
        # Exit short if price crosses above 4h HMA significantly (2% buffer)
        if in_position and position_side < 0 and close[i] > hma_4h_aligned[i] * 1.02:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals