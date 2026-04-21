#!/usr/bin/env python3
"""
1h_SwingRejection_BullBear_4hTrend
Hypothesis: On 1h timeframe, use 4h trend (EMA50) for direction and look for swing rejections at 4h swing points (HH/LL) with volume confirmation. This captures institutional order flow rejection at key levels, works in bull/bear by following 4h trend, and limits trades via strict entry conditions (price rejection + volume spike + trend alignment). Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h trend: 50-period EMA ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 4h swing points: swing high (HH) and swing low (LL) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Swing high: current high > previous 3 highs and next 3 highs (using available data)
    swing_high = np.full(len(high_4h), np.nan)
    swing_low = np.full(len(low_4h), np.nan)
    
    # Calculate swing points with lookback=3
    lookback = 3
    for i in range(lookback, len(high_4h) - lookback):
        if (high_4h[i] > np.max(high_4h[i-lookback:i]) and 
            high_4h[i] > np.max(high_4h[i+1:i+lookback+1])):
            swing_high[i] = high_4h[i]
        if (low_4h[i] < np.min(low_4h[i-lookback:i]) and 
            low_4h[i] < np.min(low_4h[i+1:i+lookback+1])):
            swing_low[i] = low_4h[i]
    
    # Align swing points to 1h timeframe
    swing_high_aligned = align_htf_to_ltf(prices, df_4h, swing_high)
    swing_low_aligned = align_htf_to_ltf(prices, df_4h, swing_low)
    
    # === Volume confirmation: 20-period volume average on 1h ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(swing_high_aligned[i]) or
            np.isnan(swing_low_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_open = prices['open'].iloc[i]
        trend_4h = ema_50_4h_aligned[i]
        swing_high_level = swing_high_aligned[i]
        swing_low_level = swing_low_aligned[i]
        vol_spike = vol_ratio[i]
        
        # Bullish rejection: close near low, open much lower, long tail down
        body_size = abs(price_close - price_open)
        lower_wick = min(price_open, price_close) - prices['low'].iloc[i]
        is_bullish_rejection = (lower_wick > 2 * body_size and 
                               price_close > price_open)
        
        # Bearish rejection: close near high, open much higher, long tail up
        upper_wick = prices['high'].iloc[i] - max(price_open, price_close)
        is_bearish_rejection = (upper_wick > 2 * body_size and 
                               price_close < price_open)
        
        if position == 0:
            # Long: bullish rejection at 4h swing low + volume spike + price above 4h EMA50
            if (not np.isnan(swing_low_level) and
                price_close <= swing_low_level * 1.001 and  # near swing low
                is_bullish_rejection and
                vol_spike > 2.0 and
                price_close > trend_4h):
                signals[i] = 0.20
                position = 1
            # Short: bearish rejection at 4h swing high + volume spike + price below 4h EMA50
            elif (not np.isnan(swing_high_level) and
                  price_close >= swing_high_level * 0.999 and  # near swing high
                  is_bearish_rejection and
                  vol_spike > 2.0 and
                  price_close < trend_4h):
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit when price crosses 4h EMA50 in opposite direction
            if position == 1 and price_close < trend_4h:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > trend_4h:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_SwingRejection_BullBear_4hTrend"
timeframe = "1h"
leverage = 1.0