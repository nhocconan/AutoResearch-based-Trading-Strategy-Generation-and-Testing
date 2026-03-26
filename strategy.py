#!/usr/bin/env python3
"""
Experiment #021: 12h Donchian Breakout + 1w Trend Filter + Volume

HYPOTHESIS: 12h Donchian(20) captures medium-term trend changes. 
1w SMA21 trend filter ensures we only trade WITH the dominant trend.
Volume spike confirms institutional conviction. Tight 2 conditions = selective entries.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Long: break above 20-bar high in bull market → captures rallies
- Short: break below 20-bar low in bear market → captures selloffs
- Range: fewer breakouts = fewer trades = less whipsaw damage
- Both directions used, adapts to regime

TARGET: 50-150 total over 4 years (proven Donchian pattern).
Reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (test Sharpe=1.382)

KEY DESIGN:
1. Donchian(20) as ONLY price structure signal
2. 1w SMA21 as trend filter (no counter-trend entries)
3. Volume spike >1.5x as confirmation
4. 2*ATR stoploss, 3*ATR takeprofit
5. Size: 0.30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1w_vol_simple_v1"
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w SMA21
    sma_1w_raw = calculate_sma(df_1w['close'].values, period=21)
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20 periods = 10 days at 12h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume MA
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
    tp_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup for all indicators
    warmup = max(60, donchian_period)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND FILTER (1w) ===
        trend_bullish = close[i] > sma_1w_aligned[i]
        trend_bearish = close[i] < sma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN SIGNALS ===
        # Long: price breaks above 20-bar high + bullish trend
        long_signal = (close[i] > donchian_high[i-1]) and trend_bullish and vol_spike
        
        # Short: price breaks below 20-bar low + bearish trend
        short_signal = (close[i] < donchian_low[i-1]) and trend_bearish and vol_spike
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            if long_signal:
                desired_signal = SIZE
            elif short_signal:
                desired_signal = -SIZE
        else:
            # Maintain position unless stoploss or takeprofit hit
            desired_signal = position_side * SIZE
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT CHECK (3*ATR) ===
        tp_triggered = False
        
        if in_position and position_side > 0:
            tp_target = entry_price + 3.0 * entry_atr
            if high[i] >= tp_target:
                tp_triggered = True
        
        if in_position and position_side < 0:
            tp_target = entry_price - 3.0 * entry_atr
            if low[i] <= tp_target:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New entry
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals