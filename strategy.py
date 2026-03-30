#!/usr/bin/env python3
"""
Experiment #010: 1d Donchian(20) + 1w EMA(21) + Volume

HYPOTHESIS: Using DAILY timeframe with WEEKLY trend filter to reduce
trade frequency while maintaining edge. Weekly EMA(21) ensures positions
align with 5-week trend structure. Daily Donchian(20) provides clear
breakout structure. Volume spike confirms institutional moves.

WHY IT SHOULD WORK IN BOTH MARKETS:
- 2021-2024: Bull market → breaks above 1d Donchian with volume → longs
- 2022 crash: Price below 1w EMA → shorts on rallies to Donchian
- 2025 bear: Below 1w EMA → shorter duration shorts, smaller positions
- 1d resolution = 4-8 trades/symbol/year = low fee drag
- ATR-based stops protect against 2022-style flash crashes

TRADE COUNT ESTIMATE:
- 1d Donchian(20) breakouts: ~1-2/month/symbol = 48-96 potential
- Volume filter (>1.3x): ~60% pass = 29-58
- 1w EMA alignment: ~50% pass = 15-29/symbol
- 4yr total: ~60-115 per symbol = TARGET ZONE

SIZE: 0.25
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_1w_ema_vol_v1"
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
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w EMA for trend (call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel (20)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 50  # Need 20 for Donchian + buffer
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === ENTRY CONDITIONS ===
        # 1w EMA alignment (trend filter)
        htf_bullish = close[i] > ema_1w_aligned[i]
        htf_bearish = close[i] < ema_1w_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.3
        
        # Donchian breakout (prior bar high/low as reference)
        prev_upper = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_lower = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = (not np.isnan(prev_upper) and close[i] > prev_upper)
        bearish_breakout = (not np.isnan(prev_lower) and close[i] < prev_lower)
        
        # Minimum hold: 2 bars (avoid chop)
        min_hold = (i - entry_bar) >= 2
        
        # === EXITS ===
        if in_position:
            # ATR stoploss
            if position_side > 0:
                stop_hit = close[i] < (entry_price - 2.5 * entry_atr)
                # Take profit at 3R
                tp_hit = close[i] > (entry_price + 3.0 * entry_atr)
            else:
                stop_hit = close[i] > (entry_price + 2.5 * entry_atr)
                tp_hit = close[i] < (entry_price - 3.0 * entry_atr)
            
            # Opposite Donchian breakout (structure exit)
            if position_side > 0 and bearish_breakout:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            elif position_side < 0 and bullish_breakout:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            elif stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            elif tp_hit:
                # Take profit: reduce to half position
                signals[i] = position_side * SIZE / 2
                in_position = True  # Keep half
                # Update entry for trailing
                entry_price = close[i]
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Above 1w EMA + Donchian breakout + volume
            if htf_bullish and bullish_breakout and vol_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = SIZE
            
            # SHORT: Below 1w EMA + Donchian breakdown + volume
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