#!/usr/bin/env python3
"""
EXPERIMENT #005 - KAMA Trend + RSI Pullback with Daily Filter (12h)
====================================================================
Hypothesis: 12h KAMA(10) captures adaptive trends while daily HMA(50) 
filters major trend direction. RSI(14) pullback entries (buy dips in 
uptrend, sell rallies in downtrend) improve entry timing vs breakouts.
ATR trailing stop protects capital. 12h TF balances trade frequency 
and signal quality.

Key features:
- Primary TF: 12h (balances noise vs signal)
- HTF filter: 1d HMA(50) for major trend direction
- Entry: KAMA slope + RSI pullback (RSI<45 in uptrend, RSI>55 in downtrend)
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.30 discrete (30% of capital)
- Regime filter: Only trade when ADX(14) > 20 (trending market)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_daily_filter_12h_v1"
timeframe = "12h"
leverage = 1.0


def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if i < period:
            kama[i] = close[i]
            continue
        
        # Efficiency Ratio
        signal = abs(close[i] - close[i - period])
        noise = sum(abs(close[j] - close[j-1]) for j in range(i - period + 1, i + 1))
        
        if noise == 0:
            er = 1.0
        else:
            er = signal / noise
        
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    tr_smooth = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values
    
    plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    return adx


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    kama = calculate_kama(close, period=10)
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    
    # Calculate KAMA slope (rate of change)
    kama_slope = np.zeros(n)
    for i in range(5, n):
        kama_slope[i] = (kama[i] - kama[i-5]) / (kama[i-5] + 1e-10)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.30  # Base position size (30% of capital)
    
    # Track position state for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    min_period = 70  # Wait for daily HMA and ADX to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr[i]) or np.isnan(adx[i]) or np.isnan(kama[i]) or
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Daily trend filter
        daily_trend = 1 if close[i] > hma_1d_aligned[i] else -1
        
        # ADX filter - only trade in trending markets (ADX > 20)
        trending_market = adx[i] > 20
        
        # KAMA slope confirmation
        kama_bullish = kama_slope[i] > 0.001
        kama_bearish = kama_slope[i] < -0.001
        
        # RSI pullback entry logic
        rsi_pullback_long = rsi[i] < 45  # Buy dip in uptrend
        rsi_pullback_short = rsi[i] > 55  # Sell rally in downtrend
        
        # Determine target signal
        target_signal = 0.0
        
        if trending_market:
            if daily_trend == 1 and kama_bullish and rsi_pullback_long:
                target_signal = SIZE  # Long entry
            elif daily_trend == -1 and kama_bearish and rsi_pullback_short:
                target_signal = -SIZE  # Short entry
        
        # Stoploss logic - check BEFORE setting new signal
        stoploss_triggered = False
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                if close[i] < trailing_stop:
                    stoploss_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                if close[i] > trailing_stop:
                    stoploss_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
        else:
            # Apply signal change
            if target_signal != 0.0:
                signals[i] = target_signal
                if position_side == 0:
                    # New entry
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
            elif position_side != 0:
                # Maintain existing position
                signals[i] = SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals