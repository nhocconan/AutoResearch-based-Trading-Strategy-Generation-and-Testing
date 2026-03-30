#!/usr/bin/env python3
"""
Experiment #005: 12h Donchian Breakout + 1d EMA Trend + Volume

HYPOTHESIS: 12h timeframe with 2 conditions (breakout + 1d trend) + volume.
The previous 12h attempts failed because:
- #011: 0 trades (3 filters too strict)
- #017: 18 trades (still too strict)

This uses 2 conditions only:
1. Donchian(10) breakout on 12h (faster = more signals)
2. 1d EMA8 for trend direction
3. Volume > 1.3x avg (mild filter, not strict spike)

ESTIMATED TRADES: 
- Donchian(10) 12h: ~1 per 10 bars = ~292/year
- Volume filter cuts 40%: ~175/year  
- 1d trend filter cuts 50%: ~87/year
- Stop-loss will exit ~50% early
- Final: ~40-60/year = 160-240 total over 4 years (slightly high but OK)

WHY IT SHOULD WORK:
- 12h captures structural moves without 15m/30m noise
- 1d EMA8 aligns with daily trend (not too slow)
- Donchian(10) catches quick mean-reversion breakouts
- Works in both bull (trend following) and bear (breakout reversion)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian10_1d_ema8_vol_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 1d HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA8 for trend direction (align to 12h)
    htf_ema8 = pd.Series(df_1d['close'].values).ewm(span=8, min_periods=8, adjust=False).mean().values
    ema8_aligned = align_htf_to_ltf(prices, df_1d, htf_ema8)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(10) - faster breakout detection
    donchian_upper = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_lower = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume average (15 bars) for spike detection
    vol_ma = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
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
    
    warmup = 50  # Enough for Donchian10, ATR14, EMA8 alignment
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema8_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION: 1d EMA8 ===
        bull_trend = close[i] > ema8_aligned[i]
        bear_trend = close[i] < ema8_aligned[i]
        
        # === VOLUME CONFIRMATION (mild: 1.3x) ===
        vol_confirm = vol_ratio[i] > 1.3
        
        # === DONCHIAN BREAKOUT (use prior bar's channel) ===
        prev_high_9 = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_low_9 = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        
        # Bullish breakout: close above prior 9-bar high
        bullish_breakout = (not np.isnan(prev_high_9) and close[i] > prev_high_9)
        
        # Bearish breakout: close below prior 9-bar low
        bearish_breakout = (not np.isnan(prev_low_9) and close[i] < prev_low_9)
        
        # === MINIMUM HOLD: 2 bars to reduce fee churn ===
        min_hold_bars = (i - entry_bar) >= 2 if in_position else True
        
        # === EXITS ===
        if in_position:
            # Stop-loss: 2.5 ATR from entry
            if position_side > 0:
                stop_price = entry_price - 2.5 * entry_atr
                stop_hit = low[i] < stop_price
            else:
                stop_price = entry_price + 2.5 * entry_atr
                stop_hit = high[i] > stop_price
            
            # Trend exit: price crosses 1d EMA8
            trend_exit = (position_side > 0 and close[i] < ema8_aligned[i]) or \
                        (position_side < 0 and close[i] > ema8_aligned[i])
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            elif min_hold_bars and trend_exit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Bullish breakout + volume confirm + bull trend
            if bullish_breakout and vol_confirm and bull_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = SIZE
            
            # SHORT: Bearish breakout + volume confirm + bear trend
            elif bearish_breakout and vol_confirm and bear_trend:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = -SIZE
    
    return signals