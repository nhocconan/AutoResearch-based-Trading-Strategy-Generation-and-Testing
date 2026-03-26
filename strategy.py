#!/usr/bin/env python3
"""
Experiment #021: 12h KAMA + Williams %R Extreme + Volume Confirmation

HYPOTHESIS: KAMA (Adaptive Moving Average) smoothly captures trend direction 
while filtering noise better than EMA. Williams %R at extremes (<-80 long, >-20 short)
captures mean-reversion bounces from oversold/overbought levels within the trend.
Volume confirmation validates the bounce has institutional backing.
Combined with 1d SMA200 for trend bias, this captures high-probability bounces
in the direction of the larger trend - working in BOTH bull (more bounces trigger)
and bear (short bounces trigger when KAMA confirms downtrend).

WHY 12h: Slower than 4h = fewer but higher-quality signals = less fee drag.
KAMA and Williams %R are both "adaptive" indicators that adjust to volatility,
making them effective across different market regimes.

TARGET: 50-150 total trades over 4 years. HARD MAX: 200.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_williams_r_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=30, fast_ema=2, slow_ema=30):
    """
    Kaufman Adaptive Moving Average
    Uses ER (Efficiency Ratio) to adjust smoothing based on trend strength.
    """
    n = len(close)
    if n < slow_ema + 1:
        return np.full(n, np.nan)
    
    # Price change over period
    change = np.abs(close[period:] - close[:-period]) if period > 0 else np.abs(np.diff(close, prepend=close[0]))
    
    # Volatility (sum of absolute changes)
    volatility = np.zeros(n)
    for i in range(1, n):
        volatility[i] = volatility[i-1] + abs(close[i] - close[i-1]) if i < period else \
                         volatility[i-1] - abs(close[i-period] - close[i-period-1]) + abs(close[i] - close[i-1])
    
    # Efficiency Ratio
    er = np.zeros(n)
    valid_idx = volatility > 0
    er[valid_idx] = change[valid_idx] / volatility[valid_idx]
    er[:period] = 0
    
    # Smoothing constant
    fast_const = 2 / (fast_ema + 1)
    slow_const = 2 / (slow_ema + 1)
    const = (er * (fast_const - slow_const) + slow_const) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if np.isnan(const[i]) or const[i] != const[i]:  # handle NaN
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + const[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_williams_r(high, low, close, period=14):
    """
    Williams %R
    Oscillator measuring close relative to high-low range.
    Values: 0 to -100. <-80 = oversold, >-20 = overbought.
    """
    n = len(close)
    willr = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period:i+1])
        lowest_low = np.min(low[i-period:i+1])
        
        if highest_high - lowest_low > 0:
            willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return willr

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
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
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Local 12h indicators
    kama_30 = calculate_kama(close, period=30, fast_ema=2, slow_ema=30)
    williams_r = calculate_williams_r(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Smooth Williams %R with EMA for less noise
    williams_smooth = pd.Series(williams_r).ewm(span=5, min_periods=5, adjust=False).mean().values
    
    # Volume for confirmation (volume spike = 1.5x average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Moderate sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 220  # Need 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(kama_30[i]) or np.isnan(sma_1d_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (12h KAMA vs price) ===
        kama_bullish = close[i] > kama_30[i]
        kama_bearish = close[i] < kama_30[i]
        
        # === TREND CONFIRMATION (1d SMA200) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        price_below_1d_sma = close[i] < sma_1d_aligned[i]
        
        # === WILLIAMS %R SIGNALS ===
        willr = williams_smooth[i]
        willr_prev = williams_smooth[i-1] if i > warmup else willr
        
        # Williams %R crossing up from oversold = long signal
        # Williams %R crossing down from overbought = short signal
        willr_crossed_up = (willr > -80 and willr_prev <= -80)
        willr_crossed_down = (willr < -20 and willr_prev >= -20)
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Williams %R crossing up from oversold + price above 1d SMA (bullish trend)
            # Volume spike confirms institutional buying
            if willr_crossed_up and price_above_1d_sma:
                if vol_spike:
                    desired_signal = SIZE
                else:
                    # Still enter without volume if signal is strong
                    desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Williams %R crossing down from overbought + price below 1d SMA (bearish trend)
            if willr_crossed_down and price_below_1d_sma:
                if vol_spike:
                    desired_signal = -SIZE
                else:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR) ===
        if in_position:
            if position_side > 0:
                stop_loss = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_loss:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
            
            if position_side < 0:
                stop_loss = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_loss:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === TAKE PROFIT EXIT ===
        if in_position:
            bars_held = i - entry_bar
            
            # Exit long when Williams %R reaches overbought
            if position_side > 0 and willr > -20:
                desired_signal = 0.0
            
            # Exit short when Williams %R reaches oversold
            if position_side < 0 and willr < -80:
                desired_signal = 0.0
            
            # Take profit at 3R (risk multiple)
            if bars_held >= 4:
                if position_side > 0:
                    profit_target = entry_price + 3.0 * atr_14[entry_bar]
                    if high[i] >= profit_target:
                        desired_signal = 0.0
                
                if position_side < 0:
                    profit_target = entry_price - 3.0 * atr_14[entry_bar]
                    if low[i] <= profit_target:
                        desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_bar = i
            # else: maintain position
        else:
            in_position = False
            position_side = 0
        
        signals[i] = desired_signal
    
    return signals