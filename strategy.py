#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm_ATRStop_v1
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation provides high-probability entries in both bull and bear markets. The Camarilla levels act as dynamic support/resistance, reducing false breakouts. Designed for low trade frequency (~25-40/year) to minimize fee drag and improve generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # === 1d trend filter: 34-period EMA ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d volume average (20-period) for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d[np.isnan(vol_ma_1d)] = 1.0  # avoid division by zero
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === ATR for dynamic stoploss (14-period on 4h) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === Camarilla levels (based on previous day) ===
    # Calculate from previous 1d bar (shifted by 1 to avoid look-ahead)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    prev_close_1d[0] = close_1d[0]  # first bar uses current close as placeholder
    prev_high_1d[0] = df_1d['high'].values[0]
    prev_low_1d[0] = df_1d['low'].values[0]
    
    # Camarilla R3, S3 levels
    range_1d = prev_high_1d - prev_low_1d
    camarilla_r3 = prev_close_1d + range_1d * 1.1 / 4
    camarilla_s3 = prev_close_1d - range_1d * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        trend_1d = ema_34_1d_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        camarilla_r3 = camarilla_r3_aligned[i]
        camarilla_s3 = camarilla_s3_aligned[i]
        atr_val = atr_14[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + volume spike > 1.5 + price above 1d EMA34
            if price_close > camarilla_r3 and vol_spike > 1.5 and price_close > trend_1d:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
                highest_since_entry = price_close
            # Short: price breaks below Camarilla S3 + volume spike > 1.5 + price below 1d EMA34
            elif price_close < camarilla_s3 and vol_spike > 1.5 and price_close < trend_1d:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
                lowest_since_entry = price_close
        
        elif position != 0:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price_high)
                # Trailing stop: 2.0 * ATR below highest since entry
                if price_close < highest_since_entry - 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, price_low)
                # Trailing stop: 2.0 * ATR above lowest since entry
                if price_close > lowest_since_entry + 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm_ATRStop_v1"
timeframe = "4h"
leverage = 1.0