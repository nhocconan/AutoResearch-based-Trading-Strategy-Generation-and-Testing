#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_R1S1_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels from previous 12h candle
    prev_close = np.roll(close_12h, 1)
    prev_close[0] = np.nan
    prev_high = np.roll(high_12h, 1)
    prev_high[0] = np.nan
    prev_low = np.roll(low_12h, 1)
    prev_low[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12.0
    # R4 = C + (H - L) * 1.1 / 2
    r4 = prev_close + (prev_high - prev_low) * 1.1 / 2.0
    # S4 = C - (H - L) * 1.1 / 2
    s4 = prev_close - (prev_high - prev_low) * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe
    pivot_4h = align_htf_to_ltf(prices, df_12h, pivot)
    r1_4h = align_htf_to_ltf(prices, df_12h, r1)
    s1_4h = align_htf_to_ltf(prices, df_12h, s1)
    r4_4h = align_htf_to_ltf(prices, df_12h, r4)
    s4_4h = align_htf_to_ltf(prices, df_12h, s4)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: EMA(34) on 4h close
    ema_34 = pd.Series(close).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(pivot_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or \
           np.isnan(r4_4h[i]) or np.isnan(s4_4h[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_34[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        price_above_ema = price > ema_34[i]
        price_below_ema = price < ema_34[i]
        
        if position == 0:
            # Long: Price breaks above R1 with volume and above EMA34
            if price > r1_4h[i] and volume_confirmed and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume and below EMA34
            elif price < s1_4h[i] and volume_confirmed and price_below_ema:
                signals[i] = -0.25
                position = -1
            # Optional: Strong breakout through R4/S4 for continuation
            elif price > r4_4h[i] and volume_confirmed and price_above_ema:
                signals[i] = 0.25
                position = 1
            elif price < s4_4h[i] and volume_confirmed and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below S1 (reversal signal) OR breaks below EMA34 (trend change)
            if price < s1_4h[i] or price < ema_34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above R1 (reversal signal) OR breaks above EMA34 (trend change)
            if price > r1_4h[i] or price > ema_34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals