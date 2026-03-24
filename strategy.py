#!/usr/bin/env python3
"""
Experiment #298: 4h Primary + 1d HTF — HMA Trend + RSI Pullback + Volume Confirmation v1

Hypothesis: Simple trend-following with pullback entries works best on 4h timeframe.
Combined with 1d HMA for major trend bias and volume confirmation to filter false breakouts.

Key insights from 200+ failed experiments:
1. Complex regime detection (Choppiness, Fisher, etc.) often overfits and fails on test
2. Funding rate contrarian hasn't worked well in recent experiments (#287 Sharpe=-0.213)
3. HMA trend + RSI pullback is proven (baseline Sharpe=0.399 on 6h)
4. Volume confirmation reduces false breakouts significantly
5. 4h timeframe should generate 20-50 trades/year with proper entry conditions

Strategy Logic:
- 1d HMA(50) = major trend bias (only trade in direction)
- 4h HMA(21) = intermediate trend
- RSI(14) pullback to 40-60 zone in trend direction = entry
- Volume > 1.5x 20-bar avg = confirmation (avoid low-volume fakeouts)
- ATR(14) 2.5x stoploss from entry

Position sizing: 0.25 base, 0.30 when 1d aligned (discrete levels)
Target: Sharpe>0.40, DD>-40%, trades>=20 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_volume_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
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
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ma(volume, period=20):
    """Volume Moving Average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_hma_slope(hma, lookback=5):
    """Calculate HMA slope (positive = uptrend, negative = downtrend)"""
    n = len(hma)
    slope = np.zeros(n)
    slope[:] = np.nan
    
    for i in range(lookback, n):
        if not np.isnan(hma[i]) and not np.isnan(hma[i-lookback]):
            slope[i] = (hma[i] - hma[i-lookback]) / hma[i-lookback] * 100.0
    
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    hma_4h_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    vol_ma = calculate_volume_ma(volume, period=20)
    sma_200 = calculate_sma(close, 200)
    
    # HMA slope for trend strength
    hma_slope = calculate_hma_slope(hma_4h, lookback=5)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(250, n):
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
        
        if np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1d MAJOR TREND BIAS ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h INTERMEDIATE TREND ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        # HMA slope confirmation
        hma_slope_positive = not np.isnan(hma_slope[i]) and hma_slope[i] > 0.5
        hma_slope_negative = not np.isnan(hma_slope[i]) and hma_slope[i] < -0.5
        
        # === SMA200 FILTER (long-term trend) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 1e-10 else False
        
        # === RSI PULLBACK ZONES ===
        # Long: RSI pulled back to 40-55 in uptrend
        rsi_pullback_long = 40.0 <= rsi[i] <= 55.0
        # Short: RSI pulled back to 45-60 in downtrend
        rsi_pullback_short = 45.0 <= rsi[i] <= 60.0
        
        # RSI oversold/overbought for reversal entries
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: 1d bull + 4h bull + RSI pullback + volume
        if htf_bull and hma_bull and rsi_pullback_long:
            if volume_confirmed or above_sma200:
                desired_signal = SIZE_STRONG if hma_slope_positive else SIZE_BASE
        
        # LONG ENTRY (reversal): 1d bull + RSI oversold + above SMA200
        elif htf_bull and rsi_oversold and above_sma200:
            desired_signal = SIZE_BASE
        
        # SHORT ENTRY: 1d bear + 4h bear + RSI pullback + volume
        elif htf_bear and hma_bear and rsi_pullback_short:
            if volume_confirmed or below_sma200:
                desired_signal = -SIZE_STRONG if hma_slope_negative else -SIZE_BASE
        
        # SHORT ENTRY (reversal): 1d bear + RSI overbought + below SMA200
        elif htf_bear and rsi_overbought and below_sma200:
            desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
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
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals