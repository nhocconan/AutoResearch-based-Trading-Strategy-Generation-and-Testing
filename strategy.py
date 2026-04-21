#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Fade_1dTrend_VolumeConfirm
Hypothesis: On 6h timeframe, price tends to revert from Camarilla R3/S3 levels in ranging markets but continues beyond R4/S4 in strong trends. 
Fade at R3/S3 when 1d trend is weak (price near 1d VWAP) with volume confirmation. Breakout continuation at R4/S4 when 1d trend is strong.
Uses 1d VWAP deviation as trend filter to distinguish ranging vs trending markets. Works in both bull/bear via mean reversion in ranges and trend following in strong moves.
Target: 50-150 trades over 4 years (12-37/year). Size: 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla pivot and VWAP)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d OHLC for Camarilla pivot calculation (based on previous 1d bar) ===
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    df_1d_volume = df_1d['volume'].values
    
    # Calculate typical price and VWAP for 1d
    typical_price = (df_1d_high + df_1d_low + df_1d_close) / 3.0
    vwap_num = pd.Series(typical_price * df_1d_volume).cumsum().values
    vwap_den = pd.Series(df_1d_volume).cumsum().values
    vwap_1d = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Calculate Camarilla levels for each 1d bar
    range_1d = df_1d_high - df_1d_low
    r3_1d = df_1d_close + 1.1 * range_1d
    s3_1d = df_1d_close - 1.1 * range_1d
    r4_1d = df_1d_close + 1.382 * range_1d
    s4_1d = df_1d_close - 1.382 * range_1d
    
    # Align 1d indicators to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # === Volume confirmation (24-period average on 6h ≈ 6d) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # === ATR (20-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) 
            or np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) 
            or np.isnan(vwap_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        vwap = vwap_1d_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume_now > 1.5 * vol_avg
        
        # VWAP deviation as trend filter: |price - vwap| / vwap
        vwap_dev = abs(price - vwap) / vwap if vwap != 0 else 0
        # Strong trend if deviation > 0.02 (2%), ranging if < 0.01 (1%)
        strong_trend = vwap_dev > 0.02
        ranging_market = vwap_dev < 0.01
        
        if position == 0:
            # Fade at R3/S3 in ranging markets with volume confirmation
            fade_long = (price < r3) and (price > s3) and ranging_market and volume_confirmed
            if price > (r3 + s3) / 2:  # Upper half of range -> fade short from R3
                fade_short_signal = price > r3 and price < r3 * 1.005  # Near R3
                if fade_short_signal:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            else:  # Lower half of range -> fade long from S3
                fade_long_signal = price < s3 and price > s3 * 0.995  # Near S3
                if fade_long_signal:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
            
            # Breakout continuation at R4/S4 in strong trends
            if price > r4 and strong_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif price < s4 and strong_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.5x ATR)
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at VWAP for fade trades
            elif price < vwap and position == 1 and entry_price > vwap:  # Long fade trade
                signals[i] = 0.0
                position = 0
            # Trend continuation exit: stop and reverse if strong opposite signal
            elif price < s3 and strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.5x ATR)
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Mean reversion exit at VWAP for fade trades
            elif price > vwap and position == -1 and entry_price < vwap:  # Short fade trade
                signals[i] = 0.0
                position = 0
            # Trend continuation exit: stop and reverse if strong opposite signal
            elif price > r3 and strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Fade_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0