#!/usr/bin/env python3
"""
Experiment #119: 4h Primary + 1d HTF — HMA Trend Bias + RSI Pullback + Volume Confirm

Hypothesis: After 100+ failed experiments, the clearest pattern is:
- Complex regime filters (Choppiness, ADX, dual-regime) = 0 trades or negative Sharpe
- SIMPLE works: HTF trend bias + LTF pullback entry + volume confirm
- 4h timeframe proven to work (20-50 trades/year target)
- Connors RSI showed ETH Sharpe +0.923 in research — use simplified version
- Volume confirmation (taker_buy_ratio) adds edge without complexity

This strategy uses MINIMAL but effective filters:
1. 1d HMA(21) = major trend bias (price above/below)
2. 4h RSI(14) pullback = entry trigger (RSI<40 long, RSI>60 short)
3. Volume confirm = taker_buy_volume/volume > 0.55 for long, < 0.45 for short
4. ATR trailing stoploss (2.5x) for risk management
5. NO Choppiness, NO ADX, NO complex regime detection

Key design choices:
- Timeframe: 4h (proven to work, balances trade frequency vs fee drag)
- HTF: 1d HMA for trend bias (simple, responsive)
- RSI thresholds: 40/60 (looser than traditional 30/70, ensures trades)
- Volume filter: taker_buy_ratio confirms institutional flow
- Position size: 0.30 (30% of capital, conservative for 4h)
- Stoploss: 2.5x ATR trailing

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_vol_pullback_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average (HMA)
    Reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Weighted Moving Average helper
    def wma(series, span):
        span = int(span)
        if span < 1:
            return np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        weights /= weights.sum()
        result = np.convolve(series, weights[::-1], mode='valid')
        return np.concatenate([np.full(span - 1, np.nan), result])
    
    half = period // 2
    if half < 1:
        half = 1
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    if len(wma_half) < len(wma_full):
        min_len = min(len(wma_half), len(wma_full))
        wma_half = wma_half[-min_len:]
        wma_full = wma_full[-min_len:]
    
    diff = 2 * wma_half - wma_full
    hma = wma(diff, int(np.sqrt(period)))
    
    # Pad to match original length
    result = np.full(n, np.nan)
    start_idx = n - len(hma)
    if start_idx >= 0:
        result[start_idx:] = hma
    
    return result

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

def calculate_taker_ratio(taker_buy_volume, volume):
    """Taker buy volume ratio (institutional flow indicator)"""
    n = len(volume)
    ratio = np.zeros(n)
    ratio[:] = np.nan
    
    for i in range(n):
        if volume[i] > 1e-10:
            ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            ratio[i] = 0.5
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    taker_ratio = calculate_taker_ratio(taker_buy_vol, volume)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (conservative for 4h)
    
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
        if np.isnan(hma_4h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(taker_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        # Simple: is price above or below daily HMA?
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND (HMA slope) ===
        # Check if 4h HMA is trending in same direction
        hma_4h_bull = hma_4h[i] > hma_4h[i-5] if i >= 5 else False
        hma_4h_bear = hma_4h[i] < hma_4h[i-5] if i >= 5 else False
        
        # === RSI PULLBACK (LOOSE - ensure trades generate on all symbols) ===
        # For longs: RSI < 40 (pullback in uptrend)
        # For shorts: RSI > 60 (rally in downtrend)
        rsi_ok_long = rsi[i] < 40.0
        rsi_ok_short = rsi[i] > 60.0
        
        # === VOLUME CONFIRMATION ===
        # For longs: taker_buy_ratio > 0.55 (buying pressure)
        # For shorts: taker_buy_ratio < 0.45 (selling pressure)
        vol_ok_long = taker_ratio[i] > 0.55
        vol_ok_short = taker_ratio[i] < 0.45
        
        # === DESIRED SIGNAL ===
        # LONG: 1d bull + 4h HMA bull + RSI<40 + volume confirm
        # SHORT: 1d bear + 4h HMA bear + RSI>60 + volume confirm
        desired_signal = 0.0
        
        if htf_bull and hma_4h_bull and rsi_ok_long and vol_ok_long:
            desired_signal = SIZE
        elif htf_bear and hma_4h_bear and rsi_ok_short and vol_ok_short:
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