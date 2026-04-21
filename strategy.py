#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_ATRStop_v1
Hypothesis: 4h Camarilla R3/S3 breakouts filtered by 12h EMA34 trend and volume spike (>2x 20-period average).
Only take longs when price breaks above R3 with uptrend and volume spike; shorts when price breaks below S3 with downtrend and volume spike.
ATR-based trailing stop with 1.5x ATR. Designed for 20-50 trades/year per symbol (~80-200 total over 4 years) to minimize fee drag.
Camarilla levels provide precise intraday support/resistance; 12h trend filter ensures alignment with higher timeframe momentum.
Works in bull/bear via 12h trend alignment and volatility-adjusted stops.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla calculation, 12h for trend)
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 2 or len(df_12h) < 34:
        return np.zeros(n)
    
    # === 1d OHLC for Camarilla levels (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels using previous completed 1d bar
    # Range = high - low
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + range_1d * 1.1 / 4.0
    camarilla_s3 = close_1d - range_1d * 1.1 / 4.0
    
    # Align to 1d timeframe (use previous completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 12h EMA34 for trend filter ===
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(34, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) 
            or np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume spike: current volume > 2x 20-period average
            volume = prices['volume'].values
            vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            vol_spike = volume[i] > 2.0 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
            
            # Long conditions: price > Camarilla R3, 12h uptrend, volume spike
            long_breakout = price > camarilla_r3_aligned[i]
            long_trend = price > ema_34_12h_aligned[i]
            
            # Short conditions: price < Camarilla S3, 12h downtrend, volume spike
            short_breakout = price < camarilla_s3_aligned[i]
            short_trend = price < ema_34_12h_aligned[i]
            
            # Entry logic - ONLY enter on volume spike + trend alignment
            if long_breakout and long_trend and vol_spoke:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below Camarilla S3 (support broken)
            elif price < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above Camarilla R3 (resistance broken)
            elif price > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0