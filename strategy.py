#!/usr/bin/env python3
"""
6h_1w_vwap_reversion_v1
Strategy: 6h VWAP reversion with weekly trend filter and volume confirmation
Timeframe: 6h
Leverage: 1.0
Hypothesis: Price tends to revert to VWAP on 6h timeframe when extended beyond 1.5 ATR, but only in the direction of the weekly trend (using weekly EMA50). Volume must be above average to confirm participation. Designed to work in both bull (buy dips to VWAP in uptrend) and bear (sell rallies to VWAP in downtrend) markets by trading mean reversion within the trend. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_vwap_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, typical_price)
    
    # ATR for deviation threshold
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap[i]) or np.isnan(atr[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        
        # Deviation from VWAP in ATR units
        vwap_deviation = (price_close - vwap[i]) / atr[i] if atr[i] > 0 else 0
        
        # Trend filter: price relative to weekly EMA50
        uptrend_1w = price_close > ema_50_1w_aligned[i]
        downtrend_1w = price_close < ema_50_1w_aligned[i]
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_avg[i]
        
        # Long: price below VWAP by >1.5 ATR in uptrend with volume
        long_signal = (vwap_deviation < -1.5) and vol_confirmed and uptrend_1w
        
        # Short: price above VWAP by >1.5 ATR in downtrend with volume
        short_signal = (vwap_deviation > 1.5) and vol_confirmed and downtrend_1w
        
        # Exit when price returns to VWAP or reverses direction
        exit_long = position == 1 and (price_close >= vwap[i] or not uptrend_1w)
        exit_short = position == -1 and (price_close <= vwap[i] or not downtrend_1w)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals