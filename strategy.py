#!/usr/bin/env python3
"""
4h_Chaikin_Oscillator_RSI_Divergence
Hypothesis: Uses Chaikin Oscillator (3,10) for money flow confirmation combined with RSI divergence on 4h timeframe.
Enters long when CO > 0 and bullish RSI divergence occurs; short when CO < 0 and bearish RSI divergence.
Includes volume confirmation and ADX trend filter to avoid whipsaws.
Designed for 4h timeframe with target 20-40 trades per year to minimize fee drag while capturing meaningful momentum shifts.
Works in both bull and bear markets by following institutional money flow and divergence signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Chaikin Oscillator (3,10) - measures money flow
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    mfm = np.where((high - low) != 0, ((close - low) - (high - close)) / (high - low), 0)
    # Money Flow Volume = Money Flow Multiplier * Volume
    mfv = mfm * volume
    # ADL = cumulative sum of MFV
    adl = np.cumsum(mfv)
    # Chaikin Oscillator = EMA(3, ADL) - EMA(10, ADL)
    adl_series = pd.Series(adl)
    ema3 = adl_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10 = adl_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin = ema3 - ema10
    
    # Calculate RSI(14) for divergence detection
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss != 0, avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate ADX(14) for trend strength filter
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / np.where(tr14 != 0, tr14, 1e-10)
    di_minus = 100 * dm_minus_14 / np.where(tr14 != 0, tr14, 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) != 0, (di_plus + di_minus), 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation - volume above 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma_20
    
    # Detect RSI divergences (lookback 5 periods)
    def detect_bullish_divergence(rsi_arr, price_arr, lookback=5):
        bullish = np.zeros_like(rsi_arr, dtype=bool)
        for i in range(lookback, len(rsi_arr)):
            if np.isnan(rsi_arr[i]) or np.isnan(rsi_arr[i-lookback]):
                continue
            # Check if price made lower low but RSI made higher low
            if (price_arr[i] < price_arr[i-lookback] and 
                rsi_arr[i] > rsi_arr[i-lookback]):
                bullish[i] = True
        return bullish
    
    def detect_bearish_divergence(rsi_arr, price_arr, lookback=5):
        bearish = np.zeros_like(rsi_arr, dtype=bool)
        for i in range(lookback, len(rsi_arr)):
            if np.isnan(rsi_arr[i]) or np.isnan(rsi_arr[i-lookback]):
                continue
            # Check if price made higher high but RSI made lower high
            if (price_arr[i] > price_arr[i-lookback] and 
                rsi_arr[i] < rsi_arr[i-lookback]):
                bearish[i] = True
        return bearish
    
    bullish_div = detect_bullish_divergence(rsi, close)
    bearish_div = detect_bearish_divergence(rsi, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chaikin[i]) or np.isnan(rsi[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Filter: only trade when ADX > 20 (trending market)
        strong_trend = adx[i] > 20
        
        # Entry conditions
        long_entry = (chaikin[i] > 0) and bullish_div[i] and vol_confirm[i] and strong_trend
        short_entry = (chaikin[i] < 0) and bearish_div[i] and vol_confirm[i] and strong_trend
        
        # Exit conditions - reverse signals or loss of momentum
        long_exit = (chaikin[i] < 0) or not bullish_div[i] or not vol_confirm[i]
        short_exit = (chaikin[i] > 0) or not bearish_div[i] or not vol_confirm[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Chaikin_Oscillator_RSI_Divergence"
timeframe = "4h"
leverage = 1.0