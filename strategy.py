#!/usr/bin/env python3
"""
Experiment #024: 12h Williams %R + Choppiness + Volume + 1d Trend

HYPOTHESIS: Williams %R hitting extreme levels (-80 for longs, -20 for shorts)
combined with Choppiness Index confirming trending market (CHOP < 38.2) captures
high-probability reversal points. Volume spike confirms institutional conviction.
12h timeframe reduces noise vs lower TFs, generates 15-25 trades/year.

KEY INSIGHT: Previous Donchian strategies failed because breakouts are hard to
time. Williams %R extremes are more reliable reversal signals, especially when
combined with trend confirmation via 1d SMA200 and regime confirmation via CHOP.

TIMEFRAME: 12h primary
HTF: 1d for trend bias
TARGET: 60-150 total trades over 4 years (15-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_willr_chop_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum indicator for reversal zones"""
    n = len(close)
    willr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        if highest_high - lowest_low > 0:
            willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return willr

def calculate_choppiness_index(high, low, close, period=14):
    """Choppiness Index - regime detector
    CHOP > 61.8 = choppy/ranging (avoid)
    CHOP < 38.2 = trending (trade)
    """
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        sum_tr = 0.0
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j - 1]) if j > 0 else high[j] - low[j])
            sum_tr += tr
        
        if highest_high - lowest_low > 0:
            chop[i] = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
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
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend bias
    sma_200_1d_raw = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d_raw)
    
    # === Calculate local 12h indicators ===
    willr_14 = calculate_williams_r(high, low, close, period=14)
    chop_14 = calculate_choppiness_index(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume MA and ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(willr_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_200_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK (Choppiness) ===
        # CHOP < 38.2 = trending, OK to trade
        # CHOP > 61.8 = ranging, avoid
        is_trending = chop_14[i] < 45.0  # Slightly relaxed threshold
        
        # === TREND BIAS (1d SMA200) ===
        price_above_sma200 = close[i] > sma_200_aligned[i]
        
        # === WILLIAMS %R EXTREMES ===
        willr_val = willr_14[i]
        # -80 = deeply oversold (potential long entry)
        # -20 = deeply overbought (potential short entry)
        oversold = willr_val < -80
        overbought = willr_val > -20
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY: Williams %R oversold + trending + bullish trend ===
            if oversold and is_trending and price_above_sma200:
                if vol_spike:
                    desired_signal = SIZE
            
            # === SHORT ENTRY: Williams %R overbought + trending + bearish trend ===
            if overbought and is_trending and not price_above_sma200:
                if vol_spike:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT: Williams %R reversal or trend break ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: %R reaches overbought OR price breaks below SMA200
            if willr_val > -20:
                exit_triggered = True
            if not price_above_sma200 and close[i] < sma_200_aligned[i]:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: %R reaches oversold OR price breaks above SMA200
            if willr_val < -80:
                exit_triggered = True
            if price_above_sma200 and close[i] > sma_200_aligned[i]:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals