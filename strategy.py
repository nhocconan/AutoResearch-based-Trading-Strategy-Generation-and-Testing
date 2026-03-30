#!/usr/bin/env python3
"""
Experiment #007: 6h MFI Divergence + 1d SMA200 Trend

HYPOTHESIS: MFI (Money Flow Index) measures VOLUME-WEIGHTED money flow,
unlike RSI which only tracks price. When MFI < 20, institutional buying
pressure accumulates. Combined with 1d SMA200 trend filter, this catches
high-probability pullback reversals in both bull and bear markets.

WHY NOVEL: None of the 25 failed experiments used MFI. RSI was tried and failed.
MFI's volume component gives it an edge over pure price oscillators.

CORE LOGIC:
- LONG: MFI < 20 + price > SMA200 + price bouncing from local low
- SHORT: MFI > 80 + price < SMA200 + price rejecting from local high
- ATR(14) stoploss at 2.0x for both directions

TARGET: 75-150 total trades over 4 years = 19-37/year. HARD MAX: 250.
Signal size: 0.25
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_mfi_sma200_vol_1d_v1"
timeframe = "6h"
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

def calculate_mfi(high, low, close, volume, period=14):
    """Money Flow Index - volume-weighted RSI"""
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan)
    
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume
    
    mfi = np.full(n, np.nan)
    
    for i in range(period, n):
        positive_flow = 0.0
        negative_flow = 0.0
        
        for j in range(i - period + 1, i + 1):
            if typical_price[j] > typical_price[j - 1]:
                positive_flow += raw_money_flow[j]
            elif typical_price[j] < typical_price[j - 1]:
                negative_flow += raw_money_flow[j]
        
        if negative_flow > 0:
            money_flow_ratio = positive_flow / negative_flow
            mfi[i] = 100.0 - (100.0 / (1.0 + money_flow_ratio))
        else:
            mfi[i] = 100.0
    
    return mfi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend (major trend filter)
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    mfi_14 = calculate_mfi(high, low, close, volume, period=14)
    
    # Volume ratio (20-bar MA for confirmation)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Local low for bounce detection
    local_low_5 = pd.Series(low).rolling(window=5, min_periods=5).min().values
    local_high_5 = pd.Series(high).rolling(window=5, min_periods=5).max().values
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    stop_price = 0.0
    
    warmup = 250  # Need 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(mfi_14[i]) or np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND FILTER (1d SMA200) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        price_below_1d_sma = close[i] < sma_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.3
        
        # MFI extremes
        mfi_oversold = mfi_14[i] < 20
        mfi_overbought = mfi_14[i] > 80
        
        # Price bouncing from local low (for longs)
        bouncing_from_low = low[i] <= local_low_5[i] * 1.002
        rejecting_from_high = high[i] >= local_high_5[i] * 0.998
        
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: MFI oversold + trend up + bounce confirmation ===
            if price_above_1d_sma and mfi_oversold and (bouncing_from_low or vol_spike):
                desired_signal = SIZE
            
            # === SHORT: MFI overbought + trend down + rejection confirmation ===
            if price_below_1d_sma and mfi_overbought and (rejecting_from_high or vol_spike):
                desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === HOLD PERIOD: minimum 2 bars (12h) to avoid fee churn ===
        bars_held = i - entry_bar
        
        # === TAKE PROFIT: MFI mean reversion ===
        if in_position and bars_held >= 2:
            # Exit longs when MFI normalizes (> 50)
            if position_side > 0 and mfi_14[i] > 50:
                desired_signal = 0.0
            # Exit shorts when MFI normalizes (< 50)
            if position_side < 0 and mfi_14[i] < 50:
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
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals