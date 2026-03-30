#!/usr/bin/env python3
"""
Experiment #010: 1d Donchian Breakout + Weekly EMA21 Trend + Volume

HYPOTHESIS: Scale up to 1d timeframe for fewer, higher-quality trades.
- Donchian(20) on 1d = 1 breakout per 20-40 days → naturally 30-80 trades/year
- Weekly EMA21 for trend = higher timeframe confirmation filters noise
- Volume spike 1.5x = confirms institutional involvement
- 2.5 ATR stoploss = survives volatility spikes

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: Breakout above 20d high + volume spike + weekly EMA21 above = strong momentum entry
- Bear: Breakdown below 20d low + volume spike + weekly EMA21 below = strong short entry
- 1d timeframe = fewer trades = less fee drag = better test generalization

EXPECTED TRADES: 80-200 total over 4 years (20-50/year per symbol)
- Donchian(20) daily = ~1 breakout per 20-40 days = 90-180/year potential
- Volume spike 1.5x → reduces by ~40%
- Weekly EMA21 filter → reduces by ~30%
- Final: ~40-80 trades over 4 years (within target of 30-100)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_wkly_ema21_vol_v1"
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
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA21 for trend direction (align to 1d)
    weekly_ema21 = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema21_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema21)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20 bars)
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
    
    warmup = 60  # Enough for Donchian20, ATR14, Weekly EMA alignment
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema21_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION: Weekly EMA21 ===
        bull_trend = close[i] > ema21_aligned[i]
        bear_trend = close[i] < ema21_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        prev_donchian_high = donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else np.nan
        prev_donchian_low = donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = (not np.isnan(prev_donchian_high) and 
                           high[i] > prev_donchian_high)
        bearish_breakout = (not np.isnan(prev_donchian_low) and 
                           low[i] < prev_donchian_low)
        
        # === MINIMUM HOLD: 2 bars to reduce fee churn ===
        min_hold_passed = (i - entry_bar) >= 2 if in_position else True
        
        # === EXITS ===
        if in_position:
            # Stop-loss: 2.5 ATR from entry
            stop_price = entry_price - 2.5 * entry_atr if position_side > 0 else entry_price + 2.5 * entry_atr
            stop_hit = (position_side > 0 and low[i] < stop_price) or (position_side < 0 and high[i] > stop_price)
            
            # Trend exit: price crosses weekly EMA21
            trend_exit = (position_side > 0 and close[i] < ema21_aligned[i]) or \
                        (position_side < 0 and close[i] > ema21_aligned[i])
            
            if stop_hit or (min_hold_passed and trend_exit):
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Bullish breakout + volume spike + bull trend
            if bullish_breakout and vol_spike and bull_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = SIZE
            
            # SHORT: Bearish breakout + volume spike + bear trend
            elif bearish_breakout and vol_spike and bear_trend:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = -SIZE
    
    return signals