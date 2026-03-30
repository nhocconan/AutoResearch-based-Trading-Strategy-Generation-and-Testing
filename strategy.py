#!/usr/bin/env python3
"""
Experiment #022: 1d Donchian Breakout + Volume Spike + Weekly Trend Filter

HYPOTHESIS: Simple daily Donchian(20) breakout with volume confirmation and 
weekly EMA(50) trend filter for position direction.

WHY IT SHOULD WORK:
- 1d timeframe = ~250 bars/year, so Donchian(20) gives ~12-18 breakout opportunities/year
- Volume spike (1.8x 20-day avg) filters false breakouts
- Weekly EMA(50) as trend filter = simple, proven, 50% of year aligned
- Expected: 30-60 trades over 4 years (8-15/year) = well within target
- ATR(14) stoploss at 2.5x manages risk in both directions
- Works in bull (long above weekly EMA) and bear (short below weekly EMA)

KEY INSIGHT: Keep it SIMPLE. Complex strategies fail. Donchian + volume + weekly trend = 3 conditions.

EXPECTED TRADE COUNT: 40-80 total over 4 years (10-20/year)
- Donchian(20) breakouts on 1d: ~15/year
- Volume spike filter (1.8x): reduces by ~35% → ~10/year
- Weekly EMA trend filter: reduces by ~30% → ~7/year
- Final: ~40-60 trades = statistical validity with low fee drag
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_vol_weekly_ema_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # === HTF indicators (1w) ===
    # Weekly EMA(50) for long-term trend direction
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Weekly ATR for stoploss scaling
    weekly_atr = calculate_atr(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, period=14)
    weekly_atr_aligned = align_htf_to_ltf(prices, df_1w, weekly_atr)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20) - 20 day high/low
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume average (20 day)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # ATR(14) warmup for 1d
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # Volume spike confirmation (1.8x 20-day average)
        vol_spike = vol_ratio[i] > 1.8
        
        # Weekly trend (bullish if price above weekly EMA50)
        weekly_bull = close[i] > ema50_1w_aligned[i]
        weekly_bear = close[i] < ema50_1w_aligned[i]
        
        # === LONG ENTRY: Price breaks above 20-day high + volume spike ===
        if not in_position:
            # Bullish breakout: high exceeds prior 20-day high
            bullish_breakout = high[i] > donchian_upper[i]
            
            if bullish_breakout and vol_spike and weekly_bull:
                desired_signal = SIZE
                
            # === SHORT ENTRY: Price breaks below 20-day low + volume spike ===
            # In bear market (price below weekly EMA), look for shorts
            bearish_breakout = low[i] < donchian_lower[i]
            
            if bearish_breakout and vol_spike and weekly_bear:
                desired_signal = -SIZE
        
        # === STOPLOSS AND EXIT ===
        if in_position:
            if position_side > 0:
                # Track highest since entry
                if high[i] > highest_since_entry:
                    highest_since_entry = high[i]
                
                # Stoploss: 2.5x ATR from entry or trail from highest
                stop_dist = max(2.5 * entry_atr, 0.5 * (highest_since_entry - entry_price))
                stop_price = highest_since_entry - stop_dist
                
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if trend reverses (price crosses below weekly EMA)
                if close[i] < ema50_1w_aligned[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Track lowest since entry
                if low[i] < lowest_since_entry:
                    lowest_since_entry = low[i]
                
                # Stoploss: 2.5x ATR from entry or trail from lowest
                stop_dist = max(2.5 * entry_atr, 0.5 * (entry_price - lowest_since_entry))
                stop_price = lowest_since_entry + stop_dist
                
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if trend reverses (price crosses above weekly EMA)
                if close[i] > ema50_1w_aligned[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 2 bars to avoid fee churn on noise ===
        if in_position and (i - entry_bar) < 2:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
        
        signals[i] = desired_signal
    
    return signals