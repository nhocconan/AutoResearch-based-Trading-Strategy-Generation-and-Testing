#!/usr/bin/env python3
"""
Experiment #028: 12h RSI Mean Reversion + 1d Supertrend Regime

HYPOTHESIS: RSI at oversold (<35) during uptrend (1d Supertrend bullish)
= high-probability long mean reversion. RSI at overbought (>65) during
downtrend (1d Supertrend bearish) = short mean reversion. Supertrend's
ATR-based bands provide natural trend/range distinction without overtrading.

WHY 12h: Trade frequency ~12-35/year (ideal for 12h target). Fewer trades
than 4h = less fee drag. More than 1d = more opportunities.

WHY IT WORKS IN BULL AND BEAR: Uses dual-direction regime (Supertrend can
be bullish OR bearish). Mean reversion plays the "snap back" after RSI
extremes. In bull markets: more long than short. In bear: more short than long.
Vol spike confirms institutional exhaustion points.

TARGET: 50-150 total trades over 4 years = 12-37/year.
Signal size: 0.25 (conservative).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_rsi_reversion_supertrend_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_rsi(prices_arr, period=14):
    """Relative Strength Index"""
    n = len(prices_arr)
    if n < period + 1:
        return np.full(n, np.nan)
    
    deltas = np.diff(prices_arr, prepend=prices_arr[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Supertrend indicator
    Returns: supertrend (1 = bullish, -1 = bearish), upper_band, lower_band
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, 0.0), np.full(n, np.nan), np.full(n, np.nan)
    
    # ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # HL2
    hl2 = (high + low) / 2
    
    # Upper and lower bands
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    supertrend[0] = 1.0  # Start bullish
    
    for i in range(1, n):
        prev_close = close[i - 1]
        curr_close = close[i]
        prev_st = supertrend[i - 1]
        
        if prev_st == 1.0:  # Was bullish
            lower_band[i] = max(lower_band[i], lower_band[i - 1])
            if curr_close < lower_band[i - 1]:
                supertrend[i] = -1.0
            else:
                supertrend[i] = 1.0
        else:  # Was bearish
            upper_band[i] = min(upper_band[i], upper_band[i - 1])
            if curr_close > upper_band[i - 1]:
                supertrend[i] = 1.0
            else:
                supertrend[i] = -1.0
    
    return supertrend, upper_band, lower_band

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Supertrend for trend direction
    st_1d, _, _ = calculate_supertrend(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values,
        period=10, 
        multiplier=3.0
    )
    st_aligned = align_htf_to_ltf(prices, df_1d, st_1d)
    
    # Local 12h indicators
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    st_entry = 0  # Supertrend direction at entry
    
    warmup = 50  # Need enough for RSI(14) + volume(20)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if st_aligned[i] == 0:  # HTF not ready
            signals[i] = 0.0
            continue
        
        # RSI value
        rsi = rsi_14[i]
        
        # Daily regime
        is_bullish_1d = st_aligned[i] == 1.0
        is_bearish_1d = st_aligned[i] == -1.0
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: RSI oversold (<35) in bullish regime
            # Mean reversion: price exhausted in uptrend
            if rsi < 35 and is_bullish_1d:
                if vol_spike:  # Confirm with volume
                    desired_signal = SIZE
            
            # === SHORT: RSI overbought (>65) in bearish regime
            # Mean reversion: price exhausted in downtrend
            if rsi > 65 and is_bearish_1d:
                if vol_spike:
                    desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        if in_position:
            # Exit if regime changes
            if position_side > 0 and is_bearish_1d:
                desired_signal = 0.0
            if position_side < 0 and is_bullish_1d:
                desired_signal = 0.0
            
            # Exit on RSI mean reversion (RSI returns toward 50)
            if position_side > 0 and rsi > 55:
                desired_signal = 0.0
            if position_side < 0 and rsi < 45:
                desired_signal = 0.0
            
            # Time-based exit: hold at least 3 bars (1.5 days)
            bars_held = i - entry_bar
            if bars_held >= 3:
                # Soft exit: RSI returned halfway
                if position_side > 0 and rsi > 48:
                    desired_signal = 0.0
                if position_side < 0 and rsi < 52:
                    desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_bar = i
                st_entry = st_aligned[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals