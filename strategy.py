#!/usr/bin/env python3
"""
Experiment #005: 12h Donchian Breakout + 1d HMA Trend + Volume Confirmation

HYPOTHESIS: Simple but robust 12h breakout strategy:
- Donchian(20) on 12h = proven price channel breakout (DB: Sharpe 1.1-1.5)
- 1d HMA(50) = HTF trend filter (bull above, bear below)
- Volume spike = trade validation
- ATR(14) stoploss = risk management

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Price breaks above 20-bar high + above 1d HMA + volume spike = long
- Bear: Price breaks below 20-bar low + below 1d HMA + volume spike = short
- Range: Choppiness Index filters out non-trending = no trade

TARGET: 75-150 total trades over 4 years (18-37/year) on 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_hma_vol_v4"
timeframe = "12h"
leverage = 1.0

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

def calculate_hma(data, period):
    """Hull Moving Average"""
    n = len(data)
    if n < period:
        return np.full(n, np.nan)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    # WMA of period/2
    wma_half = pd.Series(data).rolling(window=half, min_periods=half).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
    ).values
    
    # WMA of period
    wma_full = pd.Series(data).rolling(window=period, min_periods=period).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
    ).values
    
    # 2*WMA(half) - WMA(full)
    diff = 2 * np.nan_to_num(wma_half) - np.nan_to_num(wma_full)
    
    # HMA = WMA(sqrt(period)) of the difference
    hma = pd.Series(diff).rolling(window=sqrt_n, min_periods=sqrt_n).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
    ).values
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index: measures market choppiness
    CHOP > 61.8 = ranging (no trend, stay out)
    CHOP < 38.2 = trending (good for entries)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / range_sum) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d HMA(50) for HTF trend ===
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # 1d volume for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / np.where(vol_ma_1d > 0, vol_ma_1d, 1)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20 bars = 10 days on 12h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Choppiness Index
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio on 12h
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume / np.where(vol_ma_12h > 0, vol_ma_12h, 1)
    
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
    
    warmup = 200  # Donchian(20) + HMA(50) + volume(20)
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND FILTER (1d HMA) ===
        price_above_hma = close[i] > hma_1d_aligned[i]
        price_below_hma = close[i] < hma_1d_aligned[i]
        
        # === REGIME FILTER (Choppiness) ===
        is_trending = chop[i] < 50.0  # Relaxed from 38.2 to get more trades
        is_ranging = chop[i] > 61.8
        
        # === VOLUME CONFIRMATION ===
        vol_spike_12h = vol_ratio_12h[i] > 1.5
        vol_spike_1d = vol_ratio_1d_aligned[i] > 1.2 if not np.isnan(vol_ratio_1d_aligned[i]) else False
        vol_confirm = vol_spike_12h or vol_spike_1d
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Price breaks above 20-bar high = bullish breakout
        bullish_breakout = close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1] if i > 0 else close[i] > donchian_high[i]
        # Price breaks below 20-bar low = bearish breakout
        bearish_breakout = close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1] if i > 0 else close[i] < donchian_low[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bullish breakout + above HTF HMA + trending + volume
            if bullish_breakout and price_above_hma and (is_trending or not is_ranging) and vol_confirm:
                desired_signal = SIZE
            
            # SHORT: Bearish breakout + below HTF HMA + trending + volume
            elif bearish_breakout and price_below_hma and (is_trending or not is_ranging) and vol_confirm:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if price falls below HMA
                if price_below_hma:
                    desired_signal = 0.0
                
                # Exit if ranging market
                if is_ranging and not vol_confirm:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if price rises above HMA
                if price_above_hma:
                    desired_signal = 0.0
                
                # Exit if ranging market
                if is_ranging and not vol_confirm:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
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