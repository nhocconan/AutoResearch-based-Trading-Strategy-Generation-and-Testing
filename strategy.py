#!/usr/bin/env python3
"""
Experiment #008: 12h Donchian Breakout + Volume + 1d Trend

HYPOTHESIS: 12h Donchian(20) breakout captures medium-term momentum shifts.
Volume spike confirms institutional involvement. 1d SMA50 filters counter-trend entries.
This is the EXACT pattern from DB that achieved test Sharpe 1.10-1.38 on SOL.

WHY 12h SPECIFICALLY:
- 50-150 trades/4yr target (12-37/year) = ~1 trade every 20-60 bars
- 12h has enough data for reliable signals without overtrading
- Avoids 4h overtrading problem (DB shows 4h best performers had 90-100 trades)
- Natural middle ground between 4h frequency and 1d capital efficiency

KEY DESIGN:
1. 12h Donchian(20) breakout - proven price structure
2. Volume confirmation (>1.5x 20-avg) - institutional validation  
3. 1d SMA50 trend filter - avoid fighting larger trend
4. 2*ATR stoploss - simple, consistent risk
5. Size: 0.30 (discrete)
6. NO choppiness filter (failed: too restrictive)
7. NO EMA crosses (failed: too many false signals)

DB Reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe=1.382, 95 trades)
Target: Similar structure but on 12h for fewer, higher-quality trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_1d_sma50_v1"
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

def calculate_sma(values, period):
    """Simple Moving Average"""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === 1d data for trend filter (ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 aligned to 12h
    sma_1d_raw = calculate_sma(df_1d['close'].values, 50)
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # === 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20 periods = ~10 days)
    donchian_period = 20
    upper_band = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_band = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup (need 20 bars for Donchian + 50 for 1d SMA)
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]) or np.isnan(sma_50_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND FILTER (1d SMA50) ===
        price_above_sma = close[i] > sma_50_aligned[i]
        trend_bullish = price_above_sma
        trend_bearish = not price_above_sma
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Upper band breakout
        upper_break = close[i] > upper_band[i] and close[i-1] <= upper_band[i-1] if i > 0 else close[i] > upper_band[i]
        # Lower band breakout
        lower_break = close[i] < lower_band[i] and close[i-1] >= lower_band[i-1] if i > 0 else close[i] < lower_band[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Upper Donchian breakout + bullish trend + volume
        if upper_break and trend_bullish:
            if vol_spike:
                desired_signal = SIZE
            else:
                # Still enter without vol spike but smaller confidence
                desired_signal = SIZE * 0.8  # Reduce size if no confirmation
        
        # SHORT: Lower Donchian breakout + bearish trend + volume
        if lower_break and trend_bearish:
            if vol_spike:
                desired_signal = -SIZE
            else:
                desired_signal = -SIZE * 0.8
        
        # === STOPLOSS CHECK (2*ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            if low[i] < trailing_stop:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            if high[i] > trailing_stop:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === STOP OUT IF TREND REVERSES ===
        # If we're long but price drops below 1d SMA, exit
        if in_position and position_side > 0 and not trend_bullish:
            desired_signal = 0.0
        
        # If we're short but price rises above 1d SMA, exit
        if in_position and position_side < 0 and not trend_bearish:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals