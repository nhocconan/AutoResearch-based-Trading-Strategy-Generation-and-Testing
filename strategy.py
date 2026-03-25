#!/usr/bin/env python3
"""
Experiment #1142: 4h Primary + 1d/1w HTF — Simplified Regime + HMA Trend + RSI Pullback

Hypothesis: Previous strategy (exp #1002) had too many filters causing 0 trades.
This version SIMPLIFIES entry conditions while keeping the regime-switching logic.

Key changes from exp #1002:
1. Replace complex CRSI with simple RSI(14) - more reliable, triggers more often
2. Loosen Choppiness thresholds: >50 = choppy, <40 = trending (was 55/45)
3. Remove streak_rsi and percent_rank from CRSI calculation (too restrictive)
4. Add volume confirmation filter (vol > 0.8 * vol_sma20)
5. Simplify HMA alignment checks (2 HMA instead of 3)
6. Loosen RSI entry thresholds: long RSI<45, short RSI>55 (was more extreme)

Why this should work:
- Fewer filters = more trades (critical for passing trade count requirements)
- RSI(14) is more stable than CRSI for 4h timeframe
- Volume filter ensures we only trade on meaningful moves
- Still keeps regime-switching (proven concept from literature)
- 4h timeframe naturally limits trades to 20-50/year target

Entry conditions (LOOSENED to guarantee trades):
- LONG choppy: CHOP>50 + RSI<45 + price>1w_HMA*0.92 + vol_ok
- LONG trending: CHOP<40 + price>1d_HMA>1w_HMA + RSI>40 + RSI<70 + vol_ok
- SHORT choppy: CHOP>50 + RSI>55 + price<1w_HMA*1.08 + vol_ok
- SHORT trending: CHOP<40 + price<1d_HMA<1w_HMA + RSI<60 + RSI>30 + vol_ok

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_hma_rsi_vol_simplified_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    We use 50/40 thresholds for more sensitivity
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume SMA for confirmation
    vol_sma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_sma20[i]) or vol_sma20[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Volume confirmation (must be at least 80% of 20-bar average)
        vol_ok = volume[i] >= 0.8 * vol_sma20[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_14[i] > 50.0  # Range market
        is_trending = chop_14[i] < 40.0  # Trend market
        
        # === HTF BIAS ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Strong trend alignment
        strong_bull = hma_1d_bull and hma_1w_bull and hma_1d_aligned[i] > hma_1w_aligned[i]
        strong_bear = hma_1d_bear and hma_1w_bear and hma_1d_aligned[i] < hma_1w_aligned[i]
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE, LOOSENED) ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION MODE - RSI extremes with HTF filter
            # Long when RSI oversold and price near/above weekly HMA
            if rsi_14[i] < 45.0 and close[i] > hma_1w_aligned[i] * 0.92 and vol_ok:
                desired_signal = SIZE_BASE
            # Short when RSI overbought and price near/below weekly HMA
            elif rsi_14[i] > 55.0 and close[i] < hma_1w_aligned[i] * 1.08 and vol_ok:
                desired_signal = -SIZE_BASE
            # Stronger signals at more extreme RSI
            elif rsi_14[i] < 35.0 and close[i] > hma_1w_aligned[i] * 0.90 and vol_ok:
                desired_signal = SIZE_STRONG
            elif rsi_14[i] > 65.0 and close[i] < hma_1w_aligned[i] * 1.10 and vol_ok:
                desired_signal = -SIZE_STRONG
        
        elif is_trending:
            # TREND FOLLOWING MODE - HMA alignment + RSI filter
            # Long in strong uptrend with RSI confirmation (not overbought)
            if strong_bull and rsi_14[i] > 40.0 and rsi_14[i] < 70.0 and vol_ok:
                desired_signal = SIZE_STRONG
            # Short in strong downtrend with RSI confirmation (not oversold)
            elif strong_bear and rsi_14[i] < 60.0 and rsi_14[i] > 30.0 and vol_ok:
                desired_signal = -SIZE_STRONG
            # Weaker trend signals (single HMA alignment)
            elif hma_1d_bull and hma_1w_bull and rsi_14[i] > 45.0 and rsi_14[i] < 75.0 and vol_ok:
                desired_signal = SIZE_BASE
            elif hma_1d_bear and hma_1w_bear and rsi_14[i] < 55.0 and rsi_14[i] > 25.0 and vol_ok:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals