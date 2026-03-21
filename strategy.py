#!/usr/bin/env python3
"""
EXPERIMENT #016 - Donchian Breakout + ADX Trend Strength + HTF Filter (4h primary, 1d HTF)
==========================================================================================
Hypothesis: Donchian Channel breakouts work best when combined with trend strength (ADX)
and higher timeframe alignment. Unlike failed donchian_adx_zscore_v1, this uses:
1. Pure breakout logic (no zscore mean-reversion conflict)
2. Volatility-adjusted position sizing (smaller size in high vol = lower DD)
3. ADX > 25 filter to avoid choppy breakouts
4. 1d HMA(50) for major trend alignment (only trade breakouts in trend direction)
5. Trailing stop at 2*ATR with take-profit at 3R

Why this differs from failed strategies:
- donchian_adx_zscore_v1 mixed breakout + mean-reversion (conflicting signals)
- This is pure trend-following with proper risk management
- Volatility-adjusted sizing prevents blowups during high-vol periods (2022 crash)

Key features:
- Primary TF: 4h
- HTF filter: 1d HMA(50) for major trend
- Entry: Donchian(20) breakout + ADX(14) > 25 + HTF trend alignment
- Stoploss: 2*ATR(14) trailing
- Position sizing: 0.20-0.30, volatility-adjusted
- Take profit: Reduce to half at 3R, trail stop at 1.5R
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_adx_htf_4h_1d_v2"
timeframe = "4h"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i - 1])
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
        else:
            minus_dm[i] = 0
    
    # Smooth using Wilder's method (EMA with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate DI+ and DI-
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    mask = tr_smooth > 0
    plus_di[mask] = 100 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask2 = di_sum > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bands)"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    # Fill early values with NaN
    upper[:period - 1] = np.nan
    lower[:period - 1] = np.nan
    
    return upper, lower


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Calculate volatility for position sizing (20-period std of returns)
    returns = np.diff(close) / close[:-1]
    returns = np.insert(returns, 0, 0)
    vol = pd.Series(returns).rolling(window=20, min_periods=20).std().values
    vol = np.nan_to_num(vol, nan=0.01)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MIN_SIZE = 0.15   # Minimum position size
    MAX_SIZE = 0.35   # Maximum position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(adx[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Daily trend filter (HTF) - only trade in direction of major trend
        daily_trend = 1 if close[i] > hma_1d_aligned[i] else -1
        
        # ADX trend strength filter (avoid choppy markets)
        adx_strong = adx[i] > 25
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i - 1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i - 1]  # Break below previous lower
        
        # Volatility-adjusted position sizing
        # Target vol = 0.02 (2% daily), adjust size inversely to current vol
        target_vol = 0.02
        vol_adjustment = np.clip(target_vol / (vol[i] + 0.001), 0.5, 2.0)
        current_size = np.clip(BASE_SIZE * vol_adjustment, MIN_SIZE, MAX_SIZE)
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Donchian breakout + ADX strong + Daily trend bullish
        if breakout_long and adx_strong and daily_trend == 1:
            target_signal = current_size
        
        # Short entry: Donchian breakout + ADX strong + Daily trend bearish
        elif breakout_short and adx_strong and daily_trend == -1:
            target_signal = -current_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * entry_atr
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (3R from entry, where R = 2*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 6.0 * entry_atr:  # 3R = 6*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * entry_atr
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 6.0 * entry_atr:  # 3R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 3R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                entry_atr = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                # Exit if ADX drops below 20 (trend weakening) or HTF trend reverses
                if adx[i] < 20 or (position_side == 1 and daily_trend == -1) or (position_side == -1 and daily_trend == 1):
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = current_size * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals