#!/usr/bin/env python3
"""
Experiment #026: 12h Williams %R Momentum + 1d Trend + Volume

HYPOTHESIS: Williams %R at 12h timeframe captures momentum reversals at 
extremes (-80 oversold, -20 overbought). Combined with 1d SMA200 trend 
alignment (filter trades to major trend direction) and volume confirmation,
this catches mean-reversion bounces in trends. Works in both bull markets
(long bounces from -80 with price > SMA200) and bear markets (short rallies 
to -20 with price < SMA200). 12h is slow enough to avoid overtrading while
fast enough to capture meaningful reversals.

TIMEFRAME: 12h primary
HTF: 1d SMA200 for trend filtering
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_williams_r_1d_sma200_vol_v1"
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

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(close)
    willr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high - lowest_low > 0:
            willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return willr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction
    sma_1d_200 = df_1d['close'].values
    sma_1d_200 = pd.Series(sma_1d_200).rolling(window=200, min_periods=200).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_200)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    willr_14 = calculate_williams_r(high, low, close, period=14)
    
    # Williams %R smoothed (5-period MA of williams %R for signal line)
    willr_signal = pd.Series(willr_14).rolling(window=5, min_periods=5).mean().values
    
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    bars_held = 0
    
    warmup = max(200, 50)  # Need 200 for 1d SMA200
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(willr_14[i]) or np.isnan(willr_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND FILTER (1d SMA200) ===
        bullish_trend = close[i] > sma_1d_aligned[i]
        
        # === MOMENTUM SIGNALS ===
        willr_val = willr_14[i]
        willr_prev = willr_signal[i - 1] if i > warmup else willr_14[i - 1]
        
        # Williams %R crosses up through -50 = momentum shifting bullish
        # Williams %R crosses down through -50 = momentum shifting bearish
        cross_up = (willr_signal[i] >= -50) and (willr_prev < -50)
        cross_down = (willr_signal[i] <= -50) and (willr_prev > -50)
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === ATR for stoploss sizing ===
        current_atr = atr_14[i]
        
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Williams %R crosses up from oversold + bullish trend + volume
            if cross_up and bullish_trend and vol_spike:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Williams %R crosses down from overbought + bearish trend + volume
            if cross_down and not bullish_trend and vol_spike:
                desired_signal = -SIZE
        
        # === MINIMUM HOLDING PERIOD (2 bars = 24h) ===
        if in_position and bars_held < 2:
            # Only allow exit via stoploss, not reversal
            bars_held += 1
        
        # === STOPLOSS CHECK (2.5 ATR trailing stop) ===
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
        
        # === TAKE PROFIT AT 3R ===
        tp_triggered = False
        if in_position:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit > 3.0 * entry_atr:
                    tp_triggered = True
            if position_side < 0:
                profit = entry_price - close[i]
                if profit > 3.0 * entry_atr:
                    tp_triggered = True
        
        if tp_triggered:
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
                bars_held = 0
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            else:
                # Same direction - maintain position (no churn)
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                bars_held = 0
        
        signals[i] = desired_signal
    
    return signals