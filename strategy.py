#!/usr/bin/env python3
"""
Experiment #026: 12h Donchian(24) Breakout + Choppiness + 1d SMA200

HYPOTHESIS: 12h is the sweet spot (54% keep rate from DB).
Donchian(24) on 12h = ~12 days of lookback = 80-120 trades/year.
Choppiness Index filters out ranging markets where breakouts whipsaw.
1d SMA200 provides macro trend alignment (long only in bull, short in bear).

WHY IT WORKS IN BULL AND BEAR:
- Bull: Breakout above 12h high = momentum continuation, ride the trend
- Bear: Breakdown below 12h low = short continuation, fade rallies
- Range: Choppiness > 61.8 means no trades (avoid whipsaws)

ENTRY: Close > Donchian High(24) + Choppiness < 61.8 + price > SMA200(1d)
SHORT: Close < Donchian Low(24) + Choppiness < 61.8 + price < SMA200(1d)
EXIT: Opposite signal or 3*ATR stoploss

TARGET: 60-120 total over 4 years (15-30/year). Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian24_chop_1d_sma200_v1"
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
    CHOP > 61.8 = ranging (no trend, avoid)
    CHOP < 38.2 = trending (trade breakouts)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of true range over period
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
        
        # Highest high - lowest low over period
        highest = max(high[i - period + 1:i + 1])
        lowest = min(low[i - period + 1:i + 1])
        range_sum = highest - lowest
        
        if range_sum > 1e-10:
            chop[i] = 100 * (np.log10(tr_sum) / np.log10(range_sum))
    
    return chop

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d SMA200 for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === 12h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 24 (shift by 1 to avoid look-ahead)
    dc_upper_24 = pd.Series(high).rolling(window=24, min_periods=24).max().shift(1).values
    dc_lower_24 = pd.Series(low).rolling(window=24, min_periods=24).min().shift(1).values
    
    # Choppiness Index
    chop = calculate_choppiness(high, low, close, period=14)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_bar = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Need 200 bars for 1d SMA200 alignment
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1d_aligned[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER ===
        # Only trade when CHOP < 61.8 (trending, not ranging)
        is_trending = chop[i] < 61.8
        
        # === TREND ALIGNMENT ===
        bullish_trend = close[i] > sma_1d_aligned[i]
        bearish_trend = close[i] < sma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT ===
        bullish_breakout = (close[i] > dc_upper_24[i]) if not np.isnan(dc_upper_24[i]) else False
        bearish_breakout = (close[i] < dc_lower_24[i]) if not np.isnan(dc_lower_24[i]) else False
        
        # === TRAILING STOP UPDATE ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === MIN HOLD: 1 bar (12h) to avoid immediate whipsaw ===
        min_hold = (i - entry_bar) >= 1
        
        # === STOPLOSS CHECK (3x ATR) ===
        stop_hit = False
        if in_position:
            if position_side > 0:
                # Long stoploss: trail from highest
                stop_hit = low[i] < (entry_price - 3.0 * atr_14[i])
                # Also exit if trend flips
                if min_hold and bearish_trend and not is_trending:
                    stop_hit = True
            else:
                # Short stoploss: trail from lowest
                stop_hit = high[i] > (entry_price + 3.0 * atr_14[i])
                # Also exit if trend flips
                if min_hold and bullish_trend and not is_trending:
                    stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # === NEW POSITIONS ===
        # Long: Bullish breakout + trending + uptrend
        if bullish_breakout and is_trending and bullish_trend:
            in_position = True
            position_side = 1
            entry_bar = i
            entry_price = close[i]
            highest_since_entry = high[i]
            signals[i] = SIZE
        
        # Short: Bearish breakdown + trending + downtrend
        elif bearish_breakout and is_trending and bearish_trend:
            in_position = True
            position_side = -1
            entry_bar = i
            entry_price = close[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals