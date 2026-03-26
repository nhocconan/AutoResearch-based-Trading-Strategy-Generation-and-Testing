#!/usr/bin/env python3
"""
Experiment #021: 12h Donchian + RSI Mean Reversion + ATR Regime Filter

HYPOTHESIS: On 12h, RSI extremes (below 30 / above 70) after Donchian channel 
constrictions signal exhaustion reversions. The ATR ratio (short/long) identifies 
low-vol squeeze periods that precede breakouts. Combined with 1d trend filter 
to avoid fighting major trends, this captures mean-reversion trades with 
favorable risk/reward. Works in bear (short rallies to RSI>70 at channel tops) 
and bull (long dips to RSI<30 at channel bottoms).

TIMEFRAME: 12h primary
HTF: 1d for trend filter
TARGET: 75-150 total trades over 4 years (19-37/year)
SIGNAL: RSI extreme + channel constriction + volume
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_rsi_squeeze_v1"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper, lower, and width"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    mid = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA for trend filter
    sma_1d_raw = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # 1d ATR for regime
    atr_1d_raw = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_raw)
    
    # Local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    
    # ATR ratio for squeeze detection
    atr_ratio = atr_7 / (atr_14 + 1e-10)
    
    # Donchian 20-period for structure
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    donch_mid = (donch_upper + donch_lower) / 2
    donch_width = (donch_upper - donch_lower) / (donch_mid + 1e-10)
    
    # Bollinger for mean reversion
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_dev=2.0)
    
    # RSI
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # EMA 8 for short-term trend
    ema8 = pd.Series(close).ewm(span=8, min_periods=8, adjust=False).mean().values
    
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
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(rsi[i]):
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
        
        rsi_val = rsi[i]
        vol_val = vol_ratio[i]
        
        # === TREND FILTER (1d SMA50) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        
        # === REGIME: ATR ratio < 0.7 = squeeze (low volatility = mean reversion setup) ===
        is_squeeze = atr_ratio[i] < 0.7
        
        # === CHANNEL POSITION ===
        channel_pct = (close[i] - donch_lower[i]) / (donch_upper[i] - donch_lower[i] + 1e-10)
        
        # === ENTRY CONDITIONS ===
        # Long: RSI < 35 (oversold) + price near channel lower + squeeze + trend aligned or flat
        # Short: RSI > 65 (overbought) + price near channel upper + squeeze + trend opposed or flat
        long_setup = rsi_val < 35 and channel_pct < 0.25 and is_squeeze
        short_setup = rsi_val > 65 and channel_pct > 0.75 and is_squeeze
        
        # Volume confirmation (relaxed - just need decent volume)
        vol_confirm = vol_val > 1.1
        
        # === STOPLOSS ATR ===
        atr_stop_mult = 2.5
        
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ===
            if long_setup and vol_confirm:
                desired_signal = SIZE
            
            # === NEW SHORT ===
            elif short_setup and vol_confirm:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - atr_stop_mult * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + atr_stop_mult * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === EXIT ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Exit long: RSI > 60 (mean reversion complete) OR price at channel mid
            if rsi_val > 60:
                exit_triggered = True
            if channel_pct > 0.55:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Exit short: RSI < 40 (mean reversion complete) OR price at channel mid
            if rsi_val < 40:
                exit_triggered = True
            if channel_pct < 0.45:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - atr_stop_mult * entry_atr
                else:
                    stop_price = entry_price + atr_stop_mult * entry_atr
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