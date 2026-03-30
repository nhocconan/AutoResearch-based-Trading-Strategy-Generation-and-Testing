#!/usr/bin/env python3
"""
Experiment #027: 12h Donchian(24) + Choppiness Regime + 1d SMA Trend + Volume

HYPOTHESIS: 12h timeframe provides optimal balance between signal quality and trade frequency.
Building on session best "mtf_12h_donchian_chop_1d_sma_v3" (Sharpe=0.308, 73tr).
- 12h = ~730 bars/year = ~3000 bars in 4 years
- Donchian(24) = 12-day lookback, fewer false breakouts than shorter windows
- CHOP < 50 = trending regime filter (less strict than 40 to generate more trades)
- 1d SMA(30) = HTF trend confirmation (simpler than 50)
- Volume > 1.3x MA(20) = breakout confirmation

WHY IT SHOULD WORK IN BULL AND BEAR:
- Bull (2020-2021, 2024-2025): CHOP<50 + Donchian breakout + HTF bullish = ride rallies
- Bear (2022): CHOP>50 = stay out of chop, only trade clear 12h breakouts
- Range: Higher CHOP threshold means fewer but higher-quality trades

ENTRY: CHOP < 50 + Close > Donchian High(24) + Volume > 1.3x MA(20) + Price > 1d SMA(30)
SHORT: CHOP < 50 + Close < Donchian Low(24) + Volume > 1.3x MA(20) + Price < 1d SMA(30)
EXIT: ATR 2.5x trailing stop or opposite signal

TARGET: 60-120 total over 4 years (15-30/year). Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian24_chop_1d_sma_v4"
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
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    Using 50 as threshold for slightly more trades
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of true range over period
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]))
            else:
                tr_sum += high[j] - low[j]
        
        # Highest high - lowest low over period
        highest_high = max(high[i - period + 1:i + 1])
        lowest_low = min(low[i - period + 1:i + 1])
        hl_range = highest_high - lowest_low
        
        if hl_range > 1e-10:
            chop[i] = 100 * np.log10(tr_sum / hl_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d SMA(30) for trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=30, min_periods=30).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === 12h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Donchian 24 (shift by 1 to avoid look-ahead)
    dc_upper_24 = pd.Series(high).rolling(window=24, min_periods=24).max().shift(1).values
    dc_lower_24 = pd.Series(low).rolling(window=24, min_periods=24).min().shift(1).values
    
    # Volume MA(20)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 60  # Need enough bars for all indicators
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME CHECK: CHOP < 50 for trending market ===
        is_trending = chop_14[i] < 50.0
        
        # === HTF TREND DIRECTION FROM 1d SMA ===
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT (24-period) ===
        bullish_breakout = (close[i] > dc_upper_24[i]) if not np.isnan(dc_upper_24[i]) else False
        bearish_breakout = (close[i] < dc_lower_24[i]) if not np.isnan(dc_lower_24[i]) else False
        
        # === VOLUME CONFIRMATION (1.3x) ===
        vol_ok = volume[i] > vol_ma_20[i] * 1.3 if vol_ma_20[i] > 1e-10 else False
        
        # === TRAILING STOP UPDATE ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === MIN HOLD: 2 bars (24h) to avoid immediate whipsaw ===
        min_hold = (i - entry_bar) >= 2
        
        # === STOPLOSS CHECK (ATR 2.5x trailing) ===
        stop_hit = False
        if in_position:
            if position_side > 0:
                stop_hit = low[i] < (highest_since_entry - 2.5 * atr_14[i])
            else:
                stop_hit = high[i] > (lowest_since_entry + 2.5 * atr_14[i])
            
            # Exit on trend reversal (after min hold)
            if min_hold:
                if position_side > 0 and htf_bearish:
                    stop_hit = True
                if position_side < 0 and htf_bullish:
                    stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # === NEW POSITIONS (only in trending regime) ===
        # Long: Trending market + bullish breakout + volume confirm + HTF bullish
        if is_trending and bullish_breakout and vol_ok and htf_bullish:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        
        # Short: Trending market + bearish breakdown + volume confirm + HTF bearish
        elif is_trending and bearish_breakout and vol_ok and htf_bearish:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals