#!/usr/bin/env python3
"""
Experiment #021: Williams %R + Volume + 1d EMA Trend on 12h

HYPOTHESIS: Williams %R identifies short-term reversal points at extremes (-80/+-20),
while 1d EMA50 confirms the broader trend direction and volume confirms institutional
participation. This triple confluence targets high-probability mean-reversion trades.

WHY 12h: Slower than 4h/6h = fewer but higher-quality signals. Matches the 1d trend
reference while providing enough granularity for Williams %R extremes without the noise
of lower timeframes.

WHY IT WORKS IN BULL AND BEAR:
- Bull: Buy oversold (%R<-80) bounces WITH the 1d uptrend = higher win rate
- Bear: Sell overbought (%R>-20) rallies WITH the 1d downtrend = catches tops
- Williams %R oscillates regardless of timeframe, capturing reversals in both directions

EXPECTED BEHAVIOR:
- LONG: %R < -80 (oversold) + price > 1d EMA50 + vol spike
- SHORT: %R > -20 (overbought) + price < 1d EMA50 + vol spike

TARGET: 75-150 total trades over 4 years (19-37/year). HARD MAX: 200.
Signal size: 0.30 with discrete levels.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_willr_vol_ema50_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - Overbought/Oversold indicator"""
    n = len(close)
    willr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high != lowest_low:
            willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return willr

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
    
    # 1d EMA50 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 12h indicators ===
    willr_14 = calculate_williams_r(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (20-bar SMA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
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
    bars_since_exit = 999  # Cooldown after exit
    
    warmup = 50  # Williams %R (14) + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(willr_14[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if EMA not aligned
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Update cooldown counter
        if not in_position:
            bars_since_exit += 1
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position and bars_since_exit >= 3:
            # === LONG: Williams %R oversold + bullish trend + volume spike ===
            # Only enter if we haven't just exited (avoid whipsaw)
            if price_above_1d_ema and vol_spike and willr_14[i] < -80:
                desired_signal = SIZE
            
            # === SHORT: Williams %R overbought + bearish trend + volume spike ===
            if not price_above_1d_ema and vol_spike and willr_14[i] > -20:
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
                bars_since_exit = 0
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
                bars_since_exit = 0
        
        signals[i] = desired_signal
    
    return signals