#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: 1h Camarilla R3/S3 breakouts with 4h EMA50 trend filter and volume spike (>2.0x 20-period MA) capture institutional momentum with controlled frequency. 4h EMA50 ensures alignment with medium-term trend, reducing counter-trend whipsaws in both bull and bear markets. Session filter (08-20 UTC) avoids low-liquidity hours. Target: 15-30 trades/year (60-120 over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend, 1d for Camarilla levels)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # === 4h EMA50 trend filter ===
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === Camarilla levels from prior 1-day session (HLC of previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3, S3 levels (stronger breakout signals)
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === Volume spike filter (20-period) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(vol_ratio[i]) or np.isnan(ema_4h_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        vol_spike = vol_ratio[i]
        ema_trend = ema_4h_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 + volume spike > 2.0 + 4h EMA50 uptrend (price > EMA)
            if price_close > r3 and vol_spike > 2.0 and price_close > ema_trend:
                signals[i] = 0.20
                position = 1
                entry_price = price_close
            # Short: price breaks below S3 + volume spike > 2.0 + 4h EMA50 downtrend (price < EMA)
            elif price_close < s3 and vol_spike > 2.0 and price_close < ema_trend:
                signals[i] = -0.20
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Time-based exit: 12 bars (6 hours) to limit exposure
            if position == 1:
                if i - entry_bar >= 12:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if i - entry_bar >= 12:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0