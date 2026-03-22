#!/usr/bin/env python3
"""
Experiment #070: 4h Volatility Spike Mean Reversion + 1d/1w HMA Trend Filter

Hypothesis: 4h timeframe captures volatility spikes well for mean reversion entries.
Key insight: After volatility spikes (ATR ratio > 1.8), price tends to revert.
Combined with 1d HMA for intermediate trend and 1w HMA for long-term bias.
This should work better than pure trend following on BTC/ETH which failed 64x.

Why this might work:
1. Vol spike reversion has proven edge in crypto (panic selling → bounce)
2. 1w HMA filters out counter-trend trades in strong trends
3. 4h captures enough volatility events for trade frequency
4. Discrete position sizing controls drawdown during 2022 crash

Entry conditions (LOOSENED for trade frequency - CRITICAL for Rule 9):
- Long: ATR(7)/ATR(30) > 1.5 OR RSI < 30 + trend filter
- Short: ATR(7)/ATR(30) > 1.5 OR RSI > 70 + trend filter
- Multiple entry paths to ensure 10+ trades per symbol

Position sizing: 0.25 base, 0.35 strong signal, stoploss at 2.5*ATR
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_spike_1d_1w_hma_meanrev_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF HMAs
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (CRITICAL - Rule 2, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)
    bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === VOLATILITY SPIKE DETECTION (LOOSENED threshold) ===
        atr_ratio = atr_7[i] / (atr_30[i] + 1e-10)
        vol_spike = atr_ratio > 1.5  # Lowered from 1.8 for more trades
        
        # === TREND BIAS FROM HTF ===
        # 1w HMA = long-term trend
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # 1d HMA = intermediate trend
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === PRICE POSITION ===
        near_bb_lower = close[i] <= bb_lower[i] * 1.02
        near_bb_upper = close[i] >= bb_upper[i] * 0.98
        
        # === RSI EXTREMES (LOOSENED for more trades) ===
        rsi_oversold = rsi[i] < 35  # Lowered from 30
        rsi_overbought = rsi[i] > 65  # Lowered from 70
        rsi_extreme_oversold = rsi[i] < 25
        rsi_extreme_overbought = rsi[i] > 75
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (multiple paths - LOOSENED) ===
        
        # Path 1: Vol spike + BB lower + long-term trend up
        if vol_spike and near_bb_lower and bull_trend_1w:
            if bull_trend_1d:
                new_signal = SIZE_STRONG
            else:
                new_signal = SIZE_BASE
        
        # Path 2: RSI extreme oversold + trend filter
        if rsi_extreme_oversold and bull_trend_1w:
            new_signal = SIZE_HALF
        
        # Path 3: Vol spike + RSI oversold (no BB requirement - LOOSENED)
        if vol_spike and rsi_oversold:
            if bull_trend_1d or bull_trend_1w:
                new_signal = SIZE_BASE
        
        # Path 4: Simple BB mean reversion in uptrend (LOOSENED)
        if near_bb_lower and bull_trend_1d:
            if rsi[i] < 50:
                new_signal = SIZE_BASE
        
        # Path 5: RSI oversold alone with trend confirmation
        if rsi_oversold and bull_trend_1w and bull_trend_1d:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS (multiple paths - LOOSENED) ===
        
        # Path 1: Vol spike + BB upper + long-term trend down
        if vol_spike and near_bb_upper and bear_trend_1w:
            if bear_trend_1d:
                new_signal = -SIZE_STRONG
            else:
                new_signal = -SIZE_BASE
        
        # Path 2: RSI extreme overbought + trend filter
        if rsi_extreme_overbought and bear_trend_1w:
            new_signal = -SIZE_HALF
        
        # Path 3: Vol spike + RSI overbought (no BB requirement - LOOSENED)
        if vol_spike and rsi_overbought:
            if bear_trend_1d or bear_trend_1w:
                new_signal = -SIZE_BASE
        
        # Path 4: Simple BB mean reversion in downtrend (LOOSENED)
        if near_bb_upper and bear_trend_1d:
            if rsi[i] > 50:
                new_signal = -SIZE_BASE
        
        # Path 5: RSI overbought alone with trend confirmation
        if rsi_overbought and bear_trend_1w and bear_trend_1d:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr_14[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr_14[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr_14[i] if position_side > 0 else close[i] + 2.5 * atr_14[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr_14[i] if position_side > 0 else close[i] + 2.5 * atr_14[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals