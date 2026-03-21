#!/usr/bin/env python3
"""
EXPERIMENT #006 - Donchian Breakout + HMA Trend Filter (1d)
============================================================
Hypothesis: Daily Donchian(20) breakouts capture major trend moves in crypto.
HMA(50) on daily filters direction (only long breakouts above HMA, short below).
ADX(14) confirms trending regime. ATR(14) trailing stop protects capital.
1d timeframe reduces noise and fee churn vs lower TFs.

Key features:
- Primary TF: 1d (REQUIRED for this experiment)
- HTF filter: 1d HMA(50) for trend direction (same TF, different indicator)
- Entry: Donchian(20) breakout + HMA filter + ADX > 25
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25 discrete (25% of capital)
- Regime filter: ADX(14) > 25 (strong trending market only)

Why this should work on 1d:
- Daily breakouts have higher follow-through than intraday
- HMA(50) captures multi-week trend direction
- ADX filter avoids choppy sideways markets
- Lower trade frequency = less fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_hma_adx_daily_v1"
timeframe = "1d"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
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
    """Calculate Average Directional Index"""
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


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bands)"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    # Fill initial values
    for i in range(period-1):
        upper[i] = np.max(high[:i+1])
        lower[i] = np.min(low[:i+1])
    
    return upper, lower


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Calculate all indicators ONCE before loop
    hma = calculate_hma(close, 50)
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.25  # Base position size (25% of capital - conservative for daily)
    
    # Track position state for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    
    min_period = 70  # Wait for HMA(50), ATR(14), ADX(14), Donchian(20) to stabilize
    
    for i in range(min_period, n):
        # Check for NaN or zero in any indicator
        if (np.isnan(hma[i]) or np.isnan(atr[i]) or np.isnan(adx[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            atr[i] == 0 or hma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Trend filter: price vs HMA(50)
        price_above_hma = close[i] > hma[i]
        price_below_hma = close[i] < hma[i]
        
        # ADX filter: only trade in trending markets (ADX > 25)
        trending_market = adx[i] > 25
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # Determine target signal
        target_signal = 0.0
        
        if trending_market:
            if price_above_hma and breakout_long:
                target_signal = SIZE  # Long entry
            elif price_below_hma and breakout_short:
                target_signal = -SIZE  # Short entry
        
        # Stoploss logic - check BEFORE setting new signal
        stoploss_triggered = False
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest since entry
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                if close[i] < trailing_stop:
                    stoploss_triggered = True
            else:
                # Short position - update lowest since entry
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                if close[i] > trailing_stop:
                    stoploss_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
        else:
            # Apply signal change
            if target_signal != 0.0:
                signals[i] = target_signal
                if position_side == 0:
                    # New entry
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    entry_price = close[i]
            elif position_side != 0:
                # Maintain existing position
                signals[i] = SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals