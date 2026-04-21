#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h EMA50 trend filter and 1d volume spike confirmation.
Designed for low trade frequency (~15-35/year) to minimize fee drag while capturing breakouts in both bull and bear markets.
Uses 1h primary timeframe with 4h for trend direction and 1d for volume context.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 60 or len(df_1d) < 60:
        return np.zeros(n)
    
    # === 4h trend filter: 50-period EMA ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d volume average (20-period) for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d[np.isnan(vol_ma_1d)] = 1.0  # avoid division by zero
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === Calculate Camarilla pivot points for 1h (using previous bar's OHLC) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Shift by 1 to use previous bar's OHLC for current bar's pivot calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(r1[i]) or np.isnan(s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        trend_4h = ema_50_4h_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        camarilla_r1 = r1[i]
        camarilla_s1 = s1[i]
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices.iloc[i]['open_time']).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + volume spike > 2.0 + price above 4h EMA50
            if price_close > camarilla_r1 and vol_spike > 2.0 and price_close > trend_4h:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S1 + volume spike > 2.0 + price below 4h EMA50
            elif price_close < camarilla_s1 and vol_spike > 2.0 and price_close < trend_4h:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit on opposite Camarilla level break or loss of trend
            if position == 1:
                if price_close < camarilla_s1 or price_close < trend_4h:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if price_close > camarilla_r1 or price_close > trend_4h:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0