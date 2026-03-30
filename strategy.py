#!/usr/bin/env python3
"""
Experiment #024: 1d Donchian20 Breakout + 1w SMA Trend + Volume Spike

HYPOTHESIS: Donchian(20) breakout is a proven edge across 16K+ experiments.
Price breaking above 20d high in 1w uptrend = strong momentum continuation.
Price breaking below 20d low in 1w downtrend = continuation of bear move.
Combined with volume spike confirmation and ATR stoploss.

WHY IT WORKS IN BOTH MARKETS:
- Bull 2021: Breakouts above 20d high catch large rallies
- Bear 2022: Short breakouts below 20d low catch crash continuation
- Range 2025: Fewer signals, higher quality when trend aligns with 1w

KEY DIFFERENCE FROM FAILED STRATS: 
- Simple price channel (Donchian) not stacked indicators
- 1w SMA filter eliminates countertrend trades (main cause of failure)
- Volume spike ensures institutional participation
- ATR stoploss prevents single-day gap kills

TRADE COUNT: 50-120 total over 4 years (12-30/year). Target 75-100.
Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian20_1w_sma_vol_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper lookback"""
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
    
    # === HTF: 1w SMA for macro trend (call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    sma_1w_50 = pd.Series(df_1w['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w_50)
    
    # === 1d indicators ===
    # Donchian 20 - price channel breakout
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume spike (1.5x 20d avg)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Need 20 bars for Donchian + 50 for 1w SMA
    
    for i in range(warmup, n):
        # NaN check
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # 1w trend filter
        htf_bullish = close[i] > sma_1w_aligned[i]
        htf_bearish = close[i] < sma_1w_aligned[i]
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === BREAKOUT DETECTION ===
        # Price breaks above 20d high
        bull_breakout = close[i] > dc_upper_20[i] if not np.isnan(dc_upper_20[i]) else False
        # Price breaks below 20d low
        bear_breakout = close[i] < dc_lower_20[i] if not np.isnan(dc_lower_20[i]) else False
        
        # Volume spike confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ATR TRAILING STOP (2.5x ATR) ===
        def check_atr_stop():
            if not in_position:
                return False
            if position_side > 0:
                # Long stop: price fell below highest - 2.5*ATR
                return low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                # Short stop: price rose above lowest + 2.5*ATR
                return high[i] > (lowest_since_entry + 2.5 * entry_atr)
        
        # === MINIMUM HOLD: 3 bars (3 days) to avoid noise ===
        min_hold = (i - entry_bar) >= 3
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            
            # Exit on trend reversal (1w trend flip)
            if position_side > 0 and htf_bearish and min_hold:
                stop_hit = True
            if position_side < 0 and htf_bullish and min_hold:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Breakout above 20d high + volume spike + 1w uptrend
            if bull_breakout and vol_spike and htf_bullish:
                in_position = True
                position_side = 1
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: Breakout below 20d low + volume spike + 1w downtrend
            elif bear_breakout and vol_spike and htf_bearish:
                in_position = True
                position_side = -1
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
    
    return signals