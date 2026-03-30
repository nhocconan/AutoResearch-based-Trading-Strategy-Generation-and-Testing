#!/usr/bin/env python3
"""
Experiment #024: 4h Donchian(16) + Volume Spike + 1d EMA Filter

HYPOTHESIS: Simpler is better. Use just 3 conditions:
1. Donchian(16) breakout - structural price channel
2. Volume spike 1.3x - institutional confirmation
3. 1d EMA direction - stronger HTF trend filter than 12h

WHY IT SHOULD WORK IN BOTH MARKETS:
- Donchian(16) catches breaks in both trending and ranging markets
- Volume spike confirms institutional involvement (not retail noise)
- 1d EMA is stronger than 12h - 2022 crash had clear 1d downtrend
- 2025 bear has periods of 1d EMA alignment before breakdowns
- Fewer conditions = more consistent across symbols

TRADE COUNT TARGET: 75-150 per symbol over 4 years
- Donchian(16): ~1 breakout per 14-16 bars = 4-5/month
- 1d EMA filter: ~50% pass rate = 2-3/month
- Volume spike: ~70% pass rate = 1.5-2/month
- ~18-24 trades/symbol/year = 72-96 over 4 years ✓
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian16_vol_1d_ema_v1"
timeframe = "4h"
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
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA for trend direction (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(16) - slightly faster than 20 for more trades
    donchian_upper = pd.Series(high).rolling(window=16, min_periods=16).max().shift(1).values
    donchian_lower = pd.Series(low).rolling(window=16, min_periods=16).min().shift(1).values
    
    # Volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 50  # Need enough for Donchian(16)
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === DONCHIAN BREAKOUT (prior bar's channel) ===
        bullish_breakout = close[i] > donchian_upper[i]
        bearish_breakout = close[i] < donchian_lower[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === 1d EMA TREND DIRECTION ===
        htf_bullish = close[i] > ema_1d_aligned[i]
        htf_bearish = close[i] < ema_1d_aligned[i]
        
        # === MINIMUM HOLD: 4 bars to prevent whipsaw ===
        min_hold = (i - entry_bar) >= 4
        
        # === EXITS ===
        if in_position:
            # Stop-loss: 2.5 ATR from entry
            if position_side > 0:
                stop_hit = low[i] < (entry_price - 2.5 * entry_atr)
            else:
                stop_hit = high[i] > (entry_price + 2.5 * entry_atr)
            
            # Exit on opposite breakout
            reversal_exit = (position_side > 0 and bearish_breakout and min_hold) or \
                           (position_side < 0 and bullish_breakout and min_hold)
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            elif reversal_exit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: 1d bullish + breakout + volume
            if htf_bullish and bullish_breakout and vol_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = SIZE
            
            # SHORT: 1d bearish + breakout + volume
            elif htf_bearish and bearish_breakout and vol_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
    
    return signals