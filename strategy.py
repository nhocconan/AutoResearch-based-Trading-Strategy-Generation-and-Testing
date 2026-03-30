#!/usr/bin/env python3
"""
Experiment #008: 12h Donchian(24) + Weekly EMA21 + Choppiness Regime + Volume

HYPOTHESIS: 12h Donchian breakout with weekly trend filter and choppiness regime.
- Weekly EMA21(1w) for trend direction (bias for entry)
- Choppiness Index < 61.8 to avoid range markets (reduce whipsaws in 2022/2025)
- Donchian(24) on 12h for entry signals (~30 breakouts/year, filtered to 15-25)
- Volume spike (1.5x) for confirmation
- 2.5 ATR stoploss for risk management

EXPECTED TRADES: 60-120 total over 4 years (15-30/year per symbol)
- Donchian(24) on 12h = ~30 breakouts/year
- Choppiness filter removes ~30% in range markets
- Volume spike filter removes ~30%
- Final: ~60-120 trades over 4 years (within target of 50-200)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_wkly_ema_chop_v1"
timeframe = "12h"
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - values below 38.2 = trending, above 61.8 = choppy"""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = high[i-period:i+1].max()
        lowest = low[i-period:i+1].min()
        
        if highest == lowest:
            chop[i] = 50.0
            continue
            
        sum_tr = 0.0
        for j in range(period):
            tr = max(high[i-j] - low[i-j], 
                    abs(high[i-j] - close[i-j-1]) if i-j > 0 else high[i-j] - low[i-j],
                    abs(low[i-j] - close[i-j-1]) if i-j > 0 else 0)
            sum_tr += tr
        
        chop[i] = 100 * np.log10(sum_tr / (highest - lowest)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA21 for trend direction
    weekly_ema21 = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema21_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema21)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian Channel(24) - slightly wider than 20 to reduce trades
    donchian_upper = pd.Series(high).rolling(window=24, min_periods=24).max().values
    donchian_lower = pd.Series(low).rolling(window=24, min_periods=24).min().values
    
    # Volume average (24 bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
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
    
    warmup = 72  # Enough for Donchian24, ATR14, chop, EMA21 alignment
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema21_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME: Choppiness < 61.8 (not too choppy) ===
        is_trending = chop[i] < 61.8
        
        # === TREND DIRECTION: Weekly EMA21 ===
        bull_trend = close[i] > ema21_aligned[i]
        bear_trend = close[i] < ema21_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT (use PREVIOUS bar's channel) ===
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
            # LONG: Bullish breakout + volume spike + bull trend + trending regime
            if bullish_breakout and vol_spike and bull_trend and is_trending:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = SIZE
            
            # SHORT: Bearish breakout + volume spike + bear trend + trending regime
            elif bearish_breakout and vol_spike and bear_trend and is_trending:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = -SIZE
    
    return signals