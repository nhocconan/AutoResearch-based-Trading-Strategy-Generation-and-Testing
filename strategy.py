#!/usr/bin/env python3
"""
Experiment #023: 6h TRIX Momentum + Volume Spike + 1d ATR Regime

HYPOTHESIS: TRIX(14) triple-smooths momentum, reducing whipsaws vs RSI.
When TRIX crosses zero, momentum is shifting. Combined with:
1. Volume spike (>1.5x 20-bar MA) for institutional confirmation
2. 1d ATR regime filter (ATR_ratio > 1.2) to avoid choppy markets
3. 1d HMA for trend alignment

Why it works in BOTH bull AND bear:
- Bull: TRIX crosses above 0 + vol spike + price > 1d HMA → long
- Bear: TRIX crosses below 0 + vol spike + price < 1d HMA → short
- ATR regime avoids false signals in ranging/crash periods

TIMEFRAME: 6h primary
HTF: 1d for trend (HMA) and regime (ATR ratio)
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_trix_vol_atr_regime_v1"
timeframe = "6h"
leverage = 1.0

def calculate_trix(close, period=14):
    """Triple Smoothed Exponential Moving Average (TRIX)
    TRIX = Rate of change of triple EMA
    Cross above 0 = upward momentum, cross below 0 = downward momentum
    """
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # Triple EMA
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # TRIX = percent change of triple EMA (rate of change)
    trix = np.full(n, np.nan, dtype=np.float64)
    for i in range(period * 3, n):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i-1]) and abs(ema3[i-1]) > 1e-10:
            trix[i] = ((ema3[i] / ema3[i-1]) - 1.0) * 100.0
    
    return trix

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
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ATR for regime filter (ATR(7) / ATR(30))
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    atr_7_1d = calculate_atr(high_1d, low_1d, close_1d, period=7)
    atr_30_1d = calculate_atr(high_1d, low_1d, close_1d, period=30)
    atr_ratio_1d = atr_7_1d / (atr_30_1d + 1e-10)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # 1d HMA for trend alignment
    hma_21_1d = calculate_hma(close_1d, period=21)
    hma_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate local 6h indicators
    trix = calculate_trix(close, period=14)
    
    # TRIX signal line (EMA of TRIX)
    trix_ema = pd.Series(trix).ewm(span=9, min_periods=9, adjust=False).mean().values
    
    # Previous TRIX for cross detection
    trix_prev = np.roll(trix, 1)
    trix_prev[0] = np.nan
    
    trix_signal_prev = np.roll(trix_ema, 1)
    trix_signal_prev[0] = np.nan
    
    # ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Local RSI for additional confirmation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
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
    
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(trix[i]) or np.isnan(trix_ema[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME FILTER (1d ATR ratio) ===
        # ATR_ratio > 1.2 = volatile/trending, safe to enter
        # ATR_ratio <= 1.2 = low volatility, avoid entries
        regime_volatile = atr_ratio_aligned[i] > 1.2 if not np.isnan(atr_ratio_aligned[i]) else False
        
        # === TREND ALIGNMENT (1d HMA) ===
        price_above_1d_hma = close[i] > hma_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === RSI MOMENTUM ===
        rsi_val = rsi[i]
        
        # === TRIX CROSS DETECTION ===
        # Bullish cross: TRIX crosses above signal line AND TRIX crosses above 0
        trix_cross_up = (trix[i] > trix_ema[i]) and (trix_prev[i] <= trix_signal_prev[i])
        trix_above_zero = trix[i] > 0 and (trix_prev[i] <= 0 if not np.isnan(trix_prev[i]) else False)
        
        # Bearish cross: TRIX crosses below signal line AND TRIX crosses below 0
        trix_cross_down = (trix[i] < trix_ema[i]) and (trix_prev[i] >= trix_signal_prev[i])
        trix_below_zero = trix[i] < 0 and (trix_prev[i] >= 0 if not np.isnan(trix_prev[i]) else False)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # TRIX bullish cross (above signal AND above zero) + volume + trend aligned
            if (trix_cross_up or trix_above_zero) and vol_spike:
                # Trend aligned: price above 1d HMA (bull market)
                # OR ATR volatile (may be reversal - still take long)
                if price_above_1d_hma or regime_volatile:
                    desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # TRIX bearish cross (below signal AND below zero) + volume
            if (trix_cross_down or trix_below_zero) and vol_spike:
                # Trend aligned: price below 1d HMA (bear market)
                # OR ATR volatile (may be reversal - still take short)
                if not price_above_1d_hma or regime_volatile:
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
        
        # === EXIT CONDITIONS ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Exit long: TRIX crosses down (momentum weakening)
            if trix_cross_down:
                exit_triggered = True
            # OR RSI overbought + TRIX turning
            if rsi_val > 70 and trix[i] < trix_ema[i]:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Exit short: TRIX crosses up (momentum reversing)
            if trix_cross_up:
                exit_triggered = True
            # OR RSI oversold + TRIX turning
            if rsi_val < 30 and trix[i] > trix_ema[i]:
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
                # Same direction - maintain position
                pass
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