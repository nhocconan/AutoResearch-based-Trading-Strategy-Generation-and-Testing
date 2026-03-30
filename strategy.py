#!/usr/bin/env python3
"""
Experiment #025: 1d Donchian Breakout + Weekly EMA21 Trend + Volume (1d)

HYPOTHESIS: 1d timeframe with clear, simple conditions should capture major moves.
Weekly EMA21 as trend filter (not weekly VWAP which was too complex).
Donchian(20) breakout + volume spike + ATR stoploss.

LESSON FROM FAILURES: Too many conditions = too few trades = negative Sharpe.
Simple breakout with trend confirmation is the proven pattern.

EXPECTED TRADES: 40-80 total over 4 years (10-20/year per symbol)
- Donchian(20) on 1d = break every 20-40 days = 9-18 potential/year
- Volume spike filter (1.5x) → reduces by ~40%
- Weekly EMA21 trend filter → reduces by ~30%
- Final: 10-20/year = 40-80 total = statistical validity

RATIONALE: Previous 1d attempts failed due to:
- #010: Too few trades (14) — only Donchian, no volume
- #014: Only 2 trades — RSI too extreme
- #019: 30 trades, negative Sharpe — weekly EMA50 too slow

This version: Weekly EMA21 (faster) + volume confirmation (critical).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_ema21_vol_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(data, length):
    """Hull Moving Average"""
    series = pd.Series(data)
    half_length = length // 2
    sqrt_length = int(np.sqrt(length))
    
    wma1 = series.rolling(window=length, min_periods=length).apply(
        lambda x: np.sum(np.arange(len(x)) * x) * 2 / length, raw=True
    )
    wma2 = series.rolling(window=half_length, min_periods=half_length).apply(
        lambda x: np.sum(np.arange(len(x)) * x) / half_length, raw=True
    )
    diff = 2 * wma1 - wma2
    
    hma = diff.rolling(window=sqrt_length, min_periods=sqrt_length).mean()
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly HMA(21) for trend direction
    weekly_close = df_1w['close'].values
    weekly_hma = calculate_hma(weekly_close, 21)
    weekly_hma_aligned = align_htf_to_ltf(prices, df_1w, weekly_hma)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Donchian Channel(20) - shifted by 1 to avoid look-ahead
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Local EMA21 for micro trend
    local_ema21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Volume ratio (20-day MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 60  # Need at least 60 bars for all indicators
    
    for i in range(warmup, n):
        # Skip if missing data
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(weekly_hma_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND: Weekly HMA vs Price ===
        weekly_trend_bull = close[i] > weekly_hma_aligned[i]
        weekly_trend_bear = close[i] < weekly_hma_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT (previous bar's high/low) ===
        prev_close = close[i-1]
        prev_upper = donchian_upper[i]
        prev_lower = donchian_lower[i]
        
        # Check if previous close was below upper (pre-breakout)
        pre_bull_breakout = prev_close < prev_upper
        
        # Check if previous close was above lower (pre-breakdown)
        pre_bear_breakout = prev_close > prev_lower
        
        # Current bar breaks out
        bull_breakout = close[i] > prev_upper and pre_bull_breakout
        bear_breakout = close[i] < prev_lower and pre_bear_breakout
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Breakout above Donchian high + volume + bull trend
            if bull_breakout and vol_spike and weekly_trend_bull:
                desired_signal = SIZE
            
            # SHORT: Breakdown below Donchian low + volume + bear trend
            elif bear_breakout and vol_spike and weekly_trend_bear:
                desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        if in_position:
            if position_side > 0:
                # Stop: 2.5x ATR from entry
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Trend exit: price crosses below weekly HMA
                elif close[i] < weekly_hma_aligned[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Stop: 2.5x ATR from entry
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Trend exit: price crosses above weekly HMA
                elif close[i] > weekly_hma_aligned[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 2 bars to avoid noise ===
        if in_position and (i - entry_bar) < 2:
            desired_signal = position_side * SIZE
        
        # === EXECUTE ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        
        signals[i] = desired_signal
    
    return signals