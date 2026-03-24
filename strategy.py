#!/usr/bin/env python3
"""
Experiment #109: 4h Primary + 1d HTF — HMA Trend + BB Pullback + Volume Filter

Hypothesis: The current best (mtf_4h_hma_rsi_bb_dual_1d_v1, Sharpe=0.351) proves
that HMA + Bollinger Bands + RSI works on 4h timeframe. This experiment refines it:

1. 1d HMA = major trend bias (price above/below daily HMA)
2. 4h HMA(21) = intermediate trend confirmation
3. Bollinger Band(20, 2.0) pullback entry (enter when price pulls back to middle/lower band in uptrend)
4. Volume confirmation (volume > SMA(volume, 20) * 0.8) to filter false breakouts
5. RSI loose filter (>35 for long, <65 for short) - ensures trades on all symbols
6. ATR trailing stoploss (2.5x) for risk management

Key improvements over #101:
- HMA instead of KAMA (proven in current best strategy)
- BB pullback entry instead of simple crossover (better risk/reward)
- Volume filter to reduce false signals
- Still loose RSI to ensure trade generation

Position sizing: 0.30 (30% of capital, discrete levels)
Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_bb_vol_pullback_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    More responsive than EMA with less lag
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # WMA helper
    def wma(series, span):
        weights = np.arange(1, span + 1)
        weights = weights / weights.sum()
        result = np.convolve(series, weights, mode='valid')
        return result
    
    close_arr = close.astype(float)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA(n/2)
    wma_half = wma(close_arr, half_period)
    # WMA(n)
    wma_full = wma(close_arr, period)
    
    # Align lengths
    offset = period - sqrt_period
    
    # 2*WMA(n/2) - WMA(n)
    if len(wma_half) >= len(wma_full):
        diff = 2 * wma_half[offset:offset+len(wma_full)] - wma_full
    else:
        diff = 2 * wma_half - wma_full[:len(wma_half)]
    
    # WMA of difference with sqrt(n)
    hma_values = wma(diff, sqrt_period)
    
    # Pad with NaN
    hma = np.full(n, np.nan)
    start_idx = period - len(hma_values)
    if start_idx < 0:
        start_idx = 0
    hma[start_idx:start_idx+len(hma_values)] = hma_values
    
    return hma

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands: SMA ± std_mult * std"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Volume SMA for confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (discrete level)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(bb_middle[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND (HMA slope) ===
        # HMA above middle BB = bullish momentum
        hma_above_bb = hma_4h[i] > bb_middle[i]
        hma_below_bb = hma_4h[i] < bb_middle[i]
        
        # === VOLUME CONFIRMATION ===
        vol_ok = volume[i] > vol_sma[i] * 0.8  # At least 80% of avg volume
        
        # === RSI FILTER (LOOSE - ensure trades generate) ===
        rsi_ok_long = rsi[i] > 35.0
        rsi_ok_short = rsi[i] < 65.0
        
        # === BB PULLBACK ENTRY ===
        # Long: price pulls back to middle or lower band in uptrend
        bb_pullback_long = close[i] <= bb_middle[i] * 1.005  # Near or below middle
        bb_pullback_short = close[i] >= bb_middle[i] * 0.995  # Near or above middle
        
        # === DESIRED SIGNAL ===
        # LONG: 1d bull + 4h HMA above BB middle + volume OK + RSI > 35 + BB pullback
        # SHORT: 1d bear + 4h HMA below BB middle + volume OK + RSI < 65 + BB pullback
        desired_signal = 0.0
        
        if htf_bull and hma_above_bb and vol_ok and rsi_ok_long and bb_pullback_long:
            desired_signal = SIZE
        elif htf_bear and hma_below_bb and vol_ok and rsi_ok_short and bb_pullback_short:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals