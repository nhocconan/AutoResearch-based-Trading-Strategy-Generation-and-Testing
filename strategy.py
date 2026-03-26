#!/usr/bin/env python3
"""
Experiment #027: 1d TRIX + Choppiness Regime + Volume Confirmation

HYPOTHESIS: TRIX (Triple EMA) crossover on 1d timeframe is slow enough to 
avoid overtrading while capturing major trend reversals. Combined with:
- Choppiness Index < 50 (trending market, not ranging)
- Volume spike confirmation (institutional interest)
- ATR-based stoploss for risk management

TIMEFRAME: 1d primary
HTF: 1w for regime confirmation
TARGET: 40-80 total trades over 4 years (10-20/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_trix_chop_vol_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_trix(close, period=9):
    """TRIX - Triple EMA oscillator"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    trix = np.zeros(n, dtype=np.float64)
    trix[0] = 0
    for i in range(1, n):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i-1]) and ema3[i-1] != 0:
            trix[i] = 100 * (ema3[i] - ema3[i-1]) / ema3[i-1]
        else:
            trix[i] = 0
    
    return trix

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness
    < 38.2 = trending, > 61.8 = ranging
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        sum_tr = 0.0
        highest = -np.inf
        lowest = np.inf
        
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], 
                     abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                     abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j])
            sum_tr += tr
            if high[j] > highest:
                highest = high[j]
            if low[j] < lowest:
                lowest = low[j]
        
        range_sum = highest - lowest
        if range_sum > 0 and sum_tr > 0:
            chop[i] = 100 * np.log10(sum_tr / range_sum) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w close for regime confirmation
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w HMA for trend
    hma_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # 1w Choppiness for regime
    chop_1w = calculate_choppiness_index(high_1w, low_1w, close_1w, period=14)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # === Local 1d indicators ===
    trix_9 = calculate_trix(close, period=9)
    trix_21 = calculate_trix(close, period=21)
    
    # TRIX signal line (EMA of TRIX)
    trix_signal = pd.Series(trix_9).ewm(span=9, min_periods=9, adjust=False).mean().values
    
    # ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # 1d Choppiness for local regime
    chop_1d = calculate_choppiness_index(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # Need enough for TRIX calculation
    
    for i in range(warmup, n):
        # Check indicators available
        if np.isnan(trix_9[i]) or np.isnan(trix_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_1d[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK ===
        # Choppiness < 50 = trending (not too choppy)
        # Choppiness < 38.2 = strong trend
        local_chop = chop_1d[i]
        weekly_chop = chop_1w_aligned[i] if not np.isnan(chop_1w_aligned[i]) else 50
        
        # Both timeframes should be trending
        is_trending = (local_chop < 50) and (weekly_chop < 55)
        
        # === 1w TREND DIRECTION ===
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        trend_bullish = price_above_1w_hma
        trend_bearish = not price_above_1w_hma
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # === TRIX CROSSOVER SIGNALS ===
        # TRIX crosses above signal = bullish momentum
        # TRIX crosses below signal = bearish momentum
        trix_prev = trix_9[i-1]
        signal_prev = trix_signal[i-1]
        trix_curr = trix_9[i]
        signal_curr = trix_signal[i]
        
        # Crossover detection
        trix_cross_up = (trix_prev <= signal_prev) and (trix_curr > signal_curr)
        trix_cross_down = (trix_prev >= signal_prev) and (trix_curr < signal_curr)
        
        # TRIX momentum direction
        trix_positive = trix_curr > 0
        trix_negative = trix_curr < 0
        
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # TRIX crosses above signal + trending + volume + weekly bullish
            if trix_cross_up and is_trending and vol_spike and trend_bullish:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # TRIX crosses below signal + trending + volume + weekly bearish
            if trix_cross_down and is_trending and vol_spike and trend_bearish:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
        
        # === EXIT: Opposite signal or regime change ===
        if in_position and not is_trending and abs(desired_signal) > 0:
            # Market turned choppy - exit
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
            # else: maintain position
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                entry_bar = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals