#!/usr/bin/env python3
"""
Experiment #021: 4h TRIX Momentum + Volume Spike + Choppiness Regime

HYPOTHESIS: TRIX(14) momentum crossover is a proven momentum indicator that 
captures institutional trend changes. Combined with:
1. Volume spike confirmation (>1.5x) to filter noise
2. Choppiness Index regime filter (CHOP < 50 = trending, follow momentum)
3. ATR-based stoploss for risk management

WHY IT WORKS IN BULL AND BEAR:
- Bull: TRIX crosses positive + volume spike + price above HMA → long
- Bear: TRIX crosses negative + volume spike + price below HMA → short
- Range: CHOP > 60 = no trades (avoid whipsaws)

TIMEFRAME: 4h primary
HTF: 1d for HMA trend bias
TARGET: 75-150 total trades over 4 years (19-38/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trix_vol_chop_regime_v1"
timeframe = "4h"
leverage = 1.0

def calculate_trix(close, period=14):
    """TRIX - Triple EMA Rate of Change"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # Triple EMA
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Rate of change of triple EMA
    trix = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i - period]):
            if abs(ema3[i - period]) > 1e-10:
                trix[i] = ((ema3[i] / ema3[i - period]) - 1) * 100
    
    return trix

def calculate_signal_line(trix, signal_period=9):
    """Signal line - EMA of TRIX"""
    n = len(trix)
    signal = pd.Series(trix).ewm(span=signal_period, min_periods=signal_period, adjust=False).mean().values
    return signal

def calculate_choppiness(close, period=14):
    """Choppiness Index - lower = trending, higher = ranging"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        # Sum of true range over period
        atr_sum = 0.0
        highest = -np.inf
        lowest = np.inf
        
        for j in range(i - period + 1, i + 1):
            tr = max(close[j] - close[j-1] if j > 0 else 0, 
                     abs(close[j] - close[j-1]) if j > 0 else 0,
                     abs(close[j-1] - close[j]) if j > 0 else 0)
            atr_sum += tr if j > 0 else (close[j] - close[j]) + (high[j] - low[j])
            highest = max(highest, close[j])
            lowest = min(lowest, close[j])
        
        range_sum = highest - lowest
        if range_sum > 1e-10:
            # Simplified CHOP formula
            chop[i] = 100 * (np.log10(atr_sum / range_sum) / np.log10(period))
    
    return chop

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].values.astype(np.float64) if hasattr(series, 'values') else series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(pd.Series(close), half)
    wma_full = wma(pd.Series(close), period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(pd.Series(diff), sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    trix = calculate_trix(close, period=14)
    trix_signal = calculate_signal_line(trix, signal_period=9)
    
    # Choppiness Index
    chop = calculate_choppiness(close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(trix[i]) or np.isnan(trix_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK (Choppiness) ===
        chop_val = chop[i] if not np.isnan(chop[i]) else 50.0
        trending = chop_val < 50.0  # Below 50 = trending market
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # === MOMENTUM SIGNALS ===
        trix_val = trix[i]
        trix_sig_val = trix_signal[i]
        
        # Previous values for crossover detection
        trix_prev = trix[i - 1] if i > 0 else 0
        trix_sig_prev = trix_signal[i - 1] if i > 0 else 0
        
        # Bullish crossover: TRIX crosses above signal
        bullish_cross = (trix_val > trix_sig_val) and (trix_prev <= trix_sig_prev)
        # Bearish crossover: TRIX crosses below signal
        bearish_cross = (trix_val < trix_sig_val) and (trix_prev >= trix_sig_prev)
        
        # Momentum direction
        trix_bullish = trix_val > 0 and trix_sig_val > 0
        trix_bearish = trix_val < 0 and trix_sig_val < 0
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # === NEW LONG ENTRY ===
            # Bullish crossover + volume spike + 1d trend aligned
            if bullish_cross and vol_spike and price_above_1d_hma:
                desired_signal = SIZE
            # Alternative: strong bullish momentum in trending market
            elif trix_bullish and vol_spike and price_above_1d_hma and trending:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Bearish crossover + volume spike + 1d trend aligned
            if bearish_cross and vol_spike and not price_above_1d_hma:
                desired_signal = -SIZE
            # Alternative: strong bearish momentum in trending market
            elif trix_bearish and vol_spike and not price_above_1d_hma and trending:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT: Opposite crossover ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: bearish crossover
            if bearish_cross:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: bullish crossover
            if bullish_cross:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals