#!/usr/bin/env python3
"""
Experiment #022: Simplified Donchian-20 + 1d Trend + Volume (4h)

HYPOTHESIS: Simple price-channel breakout (proven in DB) with:
1. Donchian(20) 4h breakout - captures momentum moves
2. 1d SMA(50) for trend direction - proven HTF filter
3. Volume spike confirmation - avoids false breakouts
4. Choppiness Index - avoids range-bound whipsaws

WHY IT SHOULD WORK:
- DONCHIAN BREAKOUT is the #1 proven pattern (DB: 1.3-1.5 test Sharpe)
- SIMPLICITY = fewer trades but more reliable signals
- TARGET: 75-200 total trades (proven trade count for 4h)

KEY INSIGHT: 4h Donchian(20) = ~5 trading days. 1-2 breaks per week expected.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_1d_sma50_vol_chop_simple_v2"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(close, high, low, period=14):
    """
    Choppiness Index - detects trending vs ranging markets
    CHOP < 38.2 = trending (trend following mode)
    CHOP > 61.8 = ranging (mean reversion or no trade)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        hh = max(high[i - period + 1:i + 1])
        ll = min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / range_sum) / np.log10(period)
    
    return chop

def calculate_sma(values, period):
    """Simple moving average with min_periods"""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA(50) for HTF trend
    sma_50_1d = calculate_sma(df_1d['close'].values, 50)
    htf_above_sma = df_1d['close'].values > sma_50_1d
    htf_below_sma = df_1d['close'].values < sma_50_1d
    
    # Align HTF to 4h
    htf_above_sma_aligned = align_htf_to_ltf(prices, df_1d, htf_above_sma.astype(float))
    htf_below_sma_aligned = align_htf_to_ltf(prices, df_1d, htf_below_sma.astype(float))
    
    # === 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20 periods = 5 days on 4h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Choppiness Index
    chop = calculate_choppiness(close, high, low, period=14)
    
    # Volume ratio (20-bar average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 100  # Donchian needs 20, chop needs 14, volume needs 20
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER ===
        # Trending = chop < 50, Ranging = chop >= 50 (no new entries in ranging)
        is_trending = chop[i] < 50.0
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.4
        
        # === DONCHIAN BREAKOUT ===
        # Price breaks above 20-bar high = bullish
        bull_breakout = close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1] if i > 0 else close[i] > donchian_high[i]
        # Price breaks below 20-bar low = bearish
        bear_breakout = close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1] if i > 0 else close[i] < donchian_low[i]
        
        # === HTF TREND CONFIRMATION ===
        htf_bull = htf_above_sma_aligned[i] > 0.5 if not np.isnan(htf_above_sma_aligned[i]) else False
        htf_bear = htf_below_sma_aligned[i] > 0.5 if not np.isnan(htf_below_sma_aligned[i]) else False
        
        desired_signal = 0.0
        
        if not in_position:
            # === NEW ENTRY ===
            # LONG: bull breakout + volume confirm + HTF bull (or neutral) + trending
            if bull_breakout and vol_confirm and (htf_bull or not htf_bear):
                if is_trending:  # Only enter in trending conditions
                    desired_signal = SIZE
            
            # SHORT: bear breakout + volume confirm + HTF bear (or neutral) + trending
            elif bear_breakout and vol_confirm and (htf_bear or not htf_bull):
                if is_trending:
                    desired_signal = -SIZE
        
        else:
            # === EXISTING POSITION MANAGEMENT ===
            if position_side > 0:
                # Update trailing high
                if high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Stop: price falls more than 2.5 ATR from trail high
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                else:
                    # Price structure exit: falls below mid-channel
                    if close[i] < donchian_mid[i]:
                        desired_signal = 0.0
                    else:
                        desired_signal = SIZE
            
            elif position_side < 0:
                # Update trailing low
                if low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Stop: price rises more than 2.5 ATR from trail low
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                else:
                    # Price structure exit: rises above mid-channel
                    if close[i] > donchian_mid[i]:
                        desired_signal = 0.0
                    else:
                        desired_signal = -SIZE
        
        # === MINIMUM HOLD: 6 bars (1 day on 4h) to reduce fee churn ===
        if in_position and (i - entry_bar) < 6:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or reversal
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals