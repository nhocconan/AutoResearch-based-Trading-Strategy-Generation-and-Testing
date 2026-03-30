#!/usr/bin/env python3
"""
Experiment #021: RSI(2) + Volume Spike + SMA(200) Regime

HYPOTHESIS: Extreme RSI(2) readings (<10 or >90) with volume spikes and 
SMA(200) trend regime identify short-term reversals in both bull and bear markets.

WHY IT WORKS: RSI(2) is an ultra-short-term mean reversion indicator.
- Bull market: RSI(2) < 10 = oversold within uptrend = buy the dip
- Bear market: RSI(2) > 90 = overbought within downtrend = short the rally
- Volume spike confirms institutional participation (not retail noise)
- SMA200 provides regime direction filter

TARGET: 100-250 total trades over 4 years (25-62/year). Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_rsi2_vol_spike_sma200_v1"
timeframe = "4h"
leverage = 1.0

def calculate_rsi(prices, period=14):
    """Relative Strength Index"""
    n = len(prices)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d SMA200 for trend regime (call ONCE before loop) ===
    sma_1d_200 = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_200_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_200)
    
    # === Local indicators ===
    rsi_2 = calculate_rsi(close, period=2)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # ATR regime: short-term volatility vs long-term (trending vs ranging)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_ratio = atr_7 / np.where(atr_30 > 0, atr_30, 1)
    
    # Volume ratio (20-bar SMA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Local SMA200 for additional confirmation
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Signals
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
    entry_bar = 0
    
    warmup = 250  # Need 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(rsi_2[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1d_200_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME CHECK (1d SMA200 aligned) ===
        bull_regime = close[i] > sma_1d_200_aligned[i]
        bear_regime = close[i] < sma_1d_200_aligned[i]
        
        # ATR regime filter (avoid ranging markets)
        atr_expanding = atr_ratio[i] > 0.9  # Trending = volatility picking up
        
        # Volume spike confirmation
        vol_spike = vol_ratio[i] > 1.8
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: RSI(2) < 10 (extreme oversold) + volume spike + bull regime ===
            if bull_regime and rsi_2[i] < 10 and vol_spike and atr_expanding:
                desired_signal = SIZE
            
            # === SHORT: RSI(2) > 90 (extreme overbought) + volume spike + bear regime ===
            if bear_regime and rsi_2[i] > 90 and vol_spike and atr_expanding:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TAKE PROFIT: RSI(2) mean reversion (exit when RSI returns to 50) ===
        if in_position and position_side > 0:
            if rsi_2[i] > 50:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if rsi_2[i] < 50:
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
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals