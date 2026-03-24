#!/usr/bin/env python3
"""
Experiment #104: 4h Primary + 12h HTF — HMA Trend + BB Volatility Filter + Volume Confirmation

Hypothesis: Building on #101 success (Sharpe=0.047) but targeting current best (Sharpe=0.351):
- HMA responds faster than KAMA to trend changes (less lag, better entry timing)
- 12h HTF bias is more responsive than 1d for 4h entries (better alignment)
- Bollinger Band width filter avoids trading during extreme volatility spikes
- Volume confirmation (taker_buy_volume ratio) adds conviction filter
- Loose RSI (30/70) ensures trades generate on all symbols (BTC/ETH/SOL)
- 2.5x ATR trailing stop with position tracking

Key improvements over #101:
1. HMA instead of KAMA (faster trend detection)
2. 12h HTF instead of 1d (better responsiveness for 4h entries)
3. BB width filter (avoid extreme vol regimes)
4. Volume confirmation (taker buy ratio > 0.45 for longs, < 0.55 for shorts)
5. Position size: 0.30 (optimized for 4h timeframe)

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_bb_vol_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    def wma(series, span):
        """Weighted Moving Average"""
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            result[i] = np.sum(series[i-span+1:i+1] * weights) / np.sum(weights)
        return result
    
    close_series = pd.Series(close)
    wma_half = wma(close, period // 2)
    wma_full = wma(close, period)
    
    hma_raw = 2 * wma_half - wma_full
    hma = wma(hma_raw, int(np.sqrt(period)))
    
    return hma

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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands with width calculation"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    width = (upper - lower) / sma  # Normalized BB width
    
    return upper, lower, width

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for major trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    hma_fast = calculate_hma(close, period=16)
    hma_slow = calculate_hma(close, period=48)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Calculate BB width percentile for regime filter (rolling 100 bars)
    bb_width_pct = np.full(n, np.nan)
    for i in range(100, n):
        if not np.isnan(bb_width[i]):
            valid_widths = bb_width[i-100:i+1]
            valid_widths = valid_widths[~np.isnan(valid_widths)]
            if len(valid_widths) > 10:
                bb_width_pct[i] = np.sum(bb_width[i-100:i] < bb_width[i]) / len(valid_widths) * 100
    
    # Volume ratio (taker buy / total volume)
    vol_ratio = np.zeros(n)
    for i in range(n):
        if volume[i] > 1e-10:
            vol_ratio[i] = taker_buy_vol[i] / volume[i]
        else:
            vol_ratio[i] = 0.5
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (optimized for 4h)
    
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
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_width[i]) or np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h HMA) ===
        htf_bull = close[i] > hma_12h_aligned[i]
        htf_bear = close[i] < hma_12h_aligned[i]
        
        # === 4h TREND (HMA crossover) ===
        hma_cross_bull = hma_fast[i] > hma_slow[i]
        hma_cross_bear = hma_fast[i] < hma_slow[i]
        
        # === RSI FILTER (LOOSE - ensure trades generate) ===
        rsi_ok_long = rsi[i] > 30.0
        rsi_ok_short = rsi[i] < 70.0
        
        # === BB WIDTH FILTER (avoid extreme volatility) ===
        # Only trade when BB width is in normal range (10th-90th percentile)
        bb_ok = bb_width_pct[i] > 10.0 and bb_width_pct[i] < 90.0
        
        # === VOLUME CONFIRMATION ===
        # Long: taker buy ratio > 0.45 (buying pressure)
        # Short: taker buy ratio < 0.55 (selling pressure)
        vol_ok_long = vol_ratio[i] > 0.45
        vol_ok_short = vol_ratio[i] < 0.55
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if htf_bull and hma_cross_bull and rsi_ok_long and bb_ok and vol_ok_long:
            desired_signal = SIZE
        elif htf_bear and hma_cross_bear and rsi_ok_short and bb_ok and vol_ok_short:
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