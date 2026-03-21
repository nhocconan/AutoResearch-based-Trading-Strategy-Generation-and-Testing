#!/usr/bin/env python3
"""
EXPERIMENT #009 - HMA Trend + RSI Pullback + ADX Filter (1h)
============================================================
Hypothesis: 1h RSI pullbacks into 4h HMA trend direction capture high-probability
continuation entries when ADX confirms strong trend. This differs from previous
attempts by using ADX(14) > 25 filter to avoid choppy markets, and tighter
stoploss at 2*ATR with take-profit at 3R.

Key features:
- Primary TF: 1h (hourly candles)
- HTF filter: 4h HMA(21) for major trend direction
- Entry: RSI(14) pullback to 40-60 zone within trend
- Filter: ADX(14) > 25 ensures strong trending conditions
- Stoploss: 2.0*ATR(14) trailing
- Take profit: Reduce to half at 3R, trail stop at 1.5R
- Position sizing: 0.25-0.30 discrete levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_adx_pullback_1h_v1"
timeframe = "1h"
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


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Calculate TR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    
    # Smooth TR, +DM, -DM using Wilder's method (EMA with span=period)
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / (tr_smooth + 1e-10))
    minus_di = 100 * (minus_dm_smooth / (tr_smooth + 1e-10))
    
    # Calculate DX
    di_sum = plus_di + minus_di + 1e-10
    dx = 100 * np.abs(plus_di - minus_di) / di_sum
    
    # Calculate ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    
    # Volume moving average for confirmation
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Price relative to 4h HMA (for trend confirmation)
    price_vs_hma = (close - hma_4h_aligned) / (hma_4h_aligned + 1e-10)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.28  # Base position size (28% of capital)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(rsi[i]) or np.isnan(adx[i]) or 
            np.isnan(volume_sma[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend filter (price above/below 4h HMA)
        if close[i] > hma_4h_aligned[i]:
            htf_trend = 1  # Bullish
        else:
            htf_trend = -1  # Bearish
        
        # ADX filter - only trade when trend is strong (ADX > 25)
        trend_strong = adx[i] > 25
        
        # Volume confirmation
        volume_confirmed = volume[i] > volume_sma[i]
        
        # RSI pullback logic
        # Long: RSI pulled back to 40-55 zone in uptrend
        # Short: RSI pulled back to 45-60 zone in downtrend
        rsi_long_pullback = 40 < rsi[i] < 55
        rsi_short_pullback = 45 < rsi[i] < 60
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        if htf_trend == 1 and trend_strong:
            # Long setup: 4h bullish + strong trend + RSI pullback
            if rsi_long_pullback and volume_confirmed:
                target_signal = SIZE
        elif htf_trend == -1 and trend_strong:
            # Short setup: 4h bearish + strong trend + RSI pullback
            if rsi_short_pullback and volume_confirmed:
                target_signal = -SIZE
        
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
                
                # Check take profit (3R from entry)
                if not profit_target_hit:
                    risk = entry_price - (entry_price - 2.0 * entry_atr)  # = 2*ATR
                    if close[i] >= entry_price + 3.0 * risk / 2.0:  # 3R = 3 * 2*ATR / 2 = 3*ATR
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
                    risk = (entry_price + 2.0 * entry_atr) - entry_price  # = 2*ATR
                    if close[i] <= entry_price - 3.0 * risk / 2.0:  # 3R profit
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
                # Maintain existing position
                if not profit_target_hit:
                    signals[i] = SIZE * position_side
                else:
                    signals[i] = HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals