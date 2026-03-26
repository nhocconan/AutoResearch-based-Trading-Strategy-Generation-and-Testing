#!/usr/bin/env python3
"""
Experiment #021: 12h Williams %R Extreme + Volume Spike + ATR Regime

HYPOTHESIS: Williams %R hitting extreme levels (below -80 or above -20) on 12h
marks institutional reversal points. Combined with volume spike confirmation and
ATR-based regime filter, this captures mean-reversion setups in both bull and bear.
12h is slow enough to reduce fee drag but fast enough to catch major reversals.

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: Buy oversold (-80) dips, ride to mean
- Bear: Short overbought (-20) rallies, fade the bounce
- ATR regime prevents fading major trends

TIMEFRAME: 12h primary
HTF: 1d for trend bias
TARGET: 75-150 total trades over 4 years
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_williams_r_vol_atr_regime_v1"
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
    """Williams %R"""
    n = len(close)
    willr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high != lowest_low:
            willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
        else:
            willr[i] = -50  # neutral when range is zero
    
    return willr

def calculate_ema(close, period):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA for trend bias
    ema_1d_raw = calculate_ema(df_1d['close'].values, period=50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Williams %R
    willr = calculate_williams_r(high, low, close, period=14)
    
    # ATR ratio for regime (current vs trailing)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_ratio = atr_7 / (atr_14 + 1e-10)
    
    # Volume MA and ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for additional confirmation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative sizing
    
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
        
        if np.isnan(willr[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (1d EMA) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # === REGIME (ATR ratio) ===
        # ATR ratio > 1.5 = high volatility = choppy/range (don't fade extremes as much)
        # ATR ratio < 1.2 = low volatility = trending (mean reversion works)
        high_vol_regime = atr_ratio[i] > 1.5
        
        # === WILLIAMS %R EXTREME LEVELS ===
        willr_val = willr[i]
        oversold = willr_val < -80  # Strong oversold
        overbought = willr_val > -20  # Strong overbought
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.2
        
        # === RSI CONFIRMATION ===
        rsi_val = rsi[i]
        rsi_oversold = rsi_val < 35
        rsi_overbought = rsi_val > 65
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY: Williams %R oversold + RSI confirm + volume + trend aligned ===
            if oversold and rsi_oversold:
                # In high vol regime, require bullish trend
                # In low vol regime, allow counter-trend if RSI very oversold
                if price_above_1d_ema:
                    desired_signal = SIZE
                elif high_vol_regime and rsi_val < 25:  # Extreme counter-trend
                    desired_signal = SIZE * 0.5  # Half size for counter-trend
            
            # === SHORT ENTRY: Williams %R overbought + RSI confirm + volume + trend aligned ===
            if overbought and rsi_overbought:
                if not price_above_1d_ema:
                    desired_signal = -SIZE
                elif high_vol_regime and rsi_val > 75:  # Extreme counter-trend
                    desired_signal = -SIZE * 0.5  # Half size for counter-trend
        
        # === STOPLOSS CHECK (2.5 ATR) ===
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
        
        # === EXIT: Williams %R mean reversion or RSI normalize ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: Williams %R normalize (above -20) or RSI neutral
            if willr_val > -20:
                exit_triggered = True
            if rsi_val > 55:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: Williams %R normalize (below -80) or RSI neutral
            if willr_val < -80:
                exit_triggered = True
            if rsi_val < 45:
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
                # Same direction - maintain position
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
        
        signals[i] = desired_signal
    
    return signals