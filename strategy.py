#!/usr/bin/env python3
"""
EXPERIMENT #017 - MTF KAMA+RSI+BBWidth+Volume (15m+4h v2)
==================================================================================================
Hypothesis: Current best #004 uses Supertrend+MACD+RSI on 15m/1h/4h. This experiment tries:
- 4h KAMA trend (Kaufman Adaptive - adjusts to market noise better than HMA/DEMA)
- 15m RSI pullback entries (proven in #007, #012 kept strategies)
- Bollinger Band Width regime filter (detects squeeze vs expansion - different from ADX)
- Volume confirmation (proven in #009, #012)
- ATR trailing stop with proper signal→0 logic

Why this should beat #004 (Sharpe=3.653):
- KAMA adapts to volatility better than static EMAs (less whipsaw in choppy markets)
- BB Width regime detection filters low-volatility traps (different from ADX filter)
- 15m entries give more trade opportunities than 1h entries
- Based on lessons from #007 (Supertrend+RSI worked) and #012 (volume helped)

Key differences from failed #016:
- Uses KAMA instead of DEMA (more adaptive to market conditions)
- Uses RSI instead of Stochastic (more proven in crypto)
- Uses BB Width regime instead of ADX (different volatility measure)
- Proper numpy array handling (no read-only issues)
- Simpler position state tracking
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_rsi_bbw_volume_15m_4h_v2"
timeframe = "15m"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if volatility > 0:
            er[i] = change / volatility
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rsi = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    # Band Width = (Upper - Lower) / SMA
    bw = np.zeros(n)
    for i in range(period, n):
        if sma[i] > 0:
            bw[i] = (upper[i] - lower[i]) / sma[i]
        else:
            bw[i] = 0
    
    return upper, lower, bw


def calculate_volume_sma(volume, period=20):
    """Calculate Volume SMA for volume confirmation"""
    n = len(volume)
    if n < period:
        return np.zeros(n)
    
    volume_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return volume_sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    bb_upper_15m, bb_lower_15m, bbw_15m = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    volume_sma_15m = calculate_volume_sma(volume, period=20)
    kama_15m = calculate_kama(close, er_period=10)
    
    # Get 4h data using mtf_data helper (CRITICAL - no manual resampling!)
    try:
        df_4h = get_htf_data(prices, '4h')
        close_4h = df_4h['close'].values
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        volume_4h = df_4h['volume'].values
        
        # 4h indicators for trend
        kama_4h_raw = calculate_kama(close_4h, er_period=10)
        bbw_4h_raw = calculate_bollinger_bands(close_4h, period=20, std_mult=2.0)[2]
        atr_4h_raw = calculate_atr(high_4h, low_4h, close_4h, period=14)
        
        # Align 4h indicators to 15m timeframe (auto shift for completed bars)
        kama_4h = align_htf_to_ltf(prices, df_4h, kama_4h_raw)
        bbw_4h = align_htf_to_ltf(prices, df_4h, bbw_4h_raw)
        atr_4h = align_htf_to_ltf(prices, df_4h, atr_4h_raw)
    except Exception:
        # Fallback if mtf_data fails
        kama_4h = np.zeros(n)
        bbw_4h = np.zeros(n)
        atr_4h = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    SIZE_DYNAMIC_MIN = 0.20
    
    # RSI thresholds for entry
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 50
    RSI_SHORT_MIN = 50
    RSI_SHORT_MAX = 65
    
    # BB Width regime thresholds (percentile-based)
    BBW_LOW_PERCENTILE = 30  # Squeeze regime
    BBW_HIGH_PERCENTILE = 70  # Expansion regime
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Volume confirmation threshold
    VOLUME_MULT = 1.2
    
    # Calculate BBW percentile for regime detection
    bbw_valid = bbw_4h[100:] if len(bbw_4h) > 100 else bbw_4h
    bbw_valid = bbw_valid[bbw_valid > 0]
    bbw_low_thresh = np.percentile(bbw_valid, BBW_LOW_PERCENTILE) if len(bbw_valid) > 0 else 0.01
    bbw_high_thresh = np.percentile(bbw_valid, BBW_HIGH_PERCENTILE) if len(bbw_valid) > 0 else 0.05
    
    # ATR-based dynamic sizing baseline
    atr_valid = atr_15m[100:] if len(atr_15m) > 100 else atr_15m
    ATR_BASELINE = np.percentile(atr_valid, 50) if len(atr_valid) > 0 else np.mean(atr_valid)
    
    first_valid = max(200, 14 * 2, 20, 28)
    
    # Track position state (use lists to avoid read-only issues)
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Check for invalid data
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # 4h trend filter using KAMA
        trend_4h = 0
        if kama_4h[i] > 0 and close[i] > kama_4h[i]:
            trend_4h = 1
        elif kama_4h[i] > 0 and close[i] < kama_4h[i]:
            trend_4h = -1
        
        # BB Width regime filter (4h)
        bbw_4h_val = bbw_4h[i]
        bbw_4h_val = max(bbw_4h_val, 0.001)  # Avoid division by zero
        
        # Volume confirmation
        vol = volume[i]
        vol_sma = volume_sma_15m[i]
        volume_ok = vol_sma > 0 and vol >= vol_sma * 0.8
        
        # ATR for this bar
        atr = atr_15m[i]
        price = close[i]
        rsi = rsi_15m[i]
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = int(position_side[i - 1])
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.0*ATR)
            exit_signal = False
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    exit_signal = True
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        exit_signal = True
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    exit_signal = True
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Dynamic position sizing based on ATR volatility
        if ATR_BASELINE > 0 and atr > 0:
            atr_ratio = ATR_BASELINE / atr
            position_size = SIZE_FULL * min(max(atr_ratio, 0.5), 1.5)
            position_size = max(SIZE_DYNAMIC_MIN, min(0.40, position_size))
        else:
            position_size = SIZE_FULL
        
        # Entry logic: 4h KAMA trend + 15m RSI pullback + Volume + BBW regime
        # Only enter when BBW is in expansion regime (not squeeze)
        bbw_regime_ok = bbw_4h_val >= bbw_low_thresh
        
        if not bbw_regime_ok or not volume_ok or trend_4h == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        if trend_4h == 1:  # Bullish trend on 4h
            if RSI_LONG_MIN <= rsi <= RSI_LONG_MAX:  # RSI pullback in bullish trend
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif trend_4h == -1:  # Bearish trend on 4h
            if RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX:  # RSI pullback in bearish trend
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals