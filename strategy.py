#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeConfirm_ATRStop
Hypothesis: Camarilla R3/S3 breakout with 12h EMA34 trend filter, volume spike confirmation, and ATR trailing stop. Designed for low trade frequency (~25-35/year) to minimize fee drag and improve generalization across bull/bear markets. Uses 4h primary timeframe with 12h HTF for trend and volume context.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    # === 12h trend filter: 34-period EMA ===
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 12h volume average (20-period) for spike detection ===
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h[np.isnan(vol_ma_12h)] = 1.0  # avoid division by zero
    vol_ratio_12h = volume_12h / vol_ma_12h
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
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
    
    # === Camarilla levels (based on previous day's range) ===
    # Calculate daily high/low from 4h data
    daily_high = prices['high'].rolling(window=6, min_periods=6).max().shift(6)  # Previous day's high
    daily_low = prices['low'].rolling(window=6, min_periods=6).min().shift(6)    # Previous day's low
    daily_close = prices['close'].rolling(window=6, min_periods=6).mean().shift(6)  # Previous day's close
    
    # Camarilla R3 and S3 levels
    camarilla_range = daily_high - daily_low
    r3 = daily_close + camarilla_range * 1.1 / 4
    s3 = daily_close - camarilla_range * 1.1 / 4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(r3[i]) or np.isnan(s3[i]) or
            np.isnan(daily_high[i]) or np.isnan(daily_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        trend_12h = ema_34_12h_aligned[i]
        vol_spike = vol_ratio_12h_aligned[i]
        r3_level = r3[i]
        s3_level = s3[i]
        atr_val = atr_14[i]
        
        if position == 0:
            # Long: price breaks above R3 + volume spike > 2.0 + price above 12h EMA34
            if price_close > r3_level and vol_spike > 2.0 and price_close > trend_12h:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
                highest_since_entry = price_close
            # Short: price breaks below S3 + volume spike > 2.0 + price below 12h EMA34
            elif price_close < s3_level and vol_spike > 2.0 and price_close < trend_12h:
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

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeConfirm_ATRStop"
timeframe = "4h"
leverage = 1.0