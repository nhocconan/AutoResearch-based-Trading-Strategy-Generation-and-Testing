#!/usr/bin/env python3
"""
Experiment #021: 12h Williams %R + Volume + 1d EMA Cross Trend

HYPOTHESIS: Williams %R captures overbought/oversold extremes at multi-day scale.
By combining %R extremes (<-80 or >-20) with volume confirmation AND 1d EMA cross
trend alignment, this catches reversals at key turning points while filtering
false signals.

WHY 12h: Slower than 4h = fewer trades = less fee drag. 12h %R captures
4-8 bar swings which are more significant than hourly noise.

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Buy when %R oversold (<-80) + price above 1d SMA21 (bull trend)
- Bear: Short when %R overbought (>-20) + price below 1d SMA21 (bear trend)
- Regime-aware: Only trade in direction of 1d trend

KEY INSIGHT FROM DB: Simple indicators + volume + trend alignment > complex systems.
Williams %R is cleaner than RSI for extremes (no averaging artifact).

TARGET: 75-150 total trades over 4 years = 19-37/year. HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_willr_vol_ema_cross_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R — overbought/oversold oscillator"""
    n = len(close)
    if n < period:
        return np.full(n, -50.0)
    
    willr = np.full(n, -50.0, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = high[i - period + 1:i + 1].max()
        lowest_low = low[i - period + 1:i + 1].min()
        
        if highest_high != lowest_low:
            willr[i] = -100.0 * (highest_high - close[i]) / (highest_high - lowest_low)
        else:
            willr[i] = -50.0
    
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
    
    # 1d EMA21 for trend direction (faster than EMA50)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 12h indicators ===
    willr = calculate_williams_r(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Buffer for EMA alignment
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if 1d EMA not aligned
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA21) ===
        bull_trend = close[i] > ema_1d_aligned[i]
        bear_trend = close[i] < ema_1d_aligned[i]
        
        # Williams %R extremes
        oversold = willr[i] < -80  # Extreme oversold
        overbought = willr[i] > -20  # Extreme overbought
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: %R oversold + bull trend + volume spike ===
            if oversold and bull_trend and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: %R overbought + bear trend + volume spike ===
            if overbought and bear_trend and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR — wider for 12h swings) ===
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
        
        # === HOLD PERIOD: minimum 3 bars (1.5 days) to avoid whipsaw ===
        bars_held = i - entry_bar
        
        # === TAKE PROFIT: %R mean reversion ===
        if in_position and bars_held >= 3:
            # Long: %R crosses back above -50 (momentum shifted)
            if position_side > 0 and willr[i] > -50:
                desired_signal = 0.0
            # Short: %R crosses back below -50
            if position_side < 0 and willr[i] < -50:
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