#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1w Keltner Channels (20,2) - HTF Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # ATR(20) for Keltner
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[high_1w[0] - low_1w[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_20 = np.full_like(close_1w, np.nan)
    for i in range(len(atr_20)):
        if i < 19:
            if i == 0:
                atr_20[i] = tr[0]
            else:
                atr_20[i] = (atr_20[i-1] * 19 + tr[i]) / 20
        else:
            atr_20[i] = (atr_20[i-1] * 19 + tr[i]) / 20
    
    # Keltner Middle (EMA20)
    ema_20 = np.full_like(close_1w, np.nan)
    for i in range(len(ema_20)):
        if i == 0:
            ema_20[i] = close_1w[0]
        else:
            ema_20[i] = (close_1w[i] * 2 + ema_20[i-1] * 19) / 20
    
    upper_keltner = ema_20 + 2 * atr_20
    lower_keltner = ema_20 - 2 * atr_20
    
    # === 1d Donchian Channel (20) - Entry Signal ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian Upper/Lower (20)
    donch_high = np.full_like(high_1d, np.nan)
    donch_low = np.full_like(low_1d, np.nan)
    for i in range(len(donch_high)):
        if i < 19:
            if i == 0:
                donch_high[i] = high_1d[0]
                donch_low[i] = low_1d[0]
            else:
                donch_high[i] = np.max(high_1d[:i+1])
                donch_low[i] = np.min(low_1d[:i+1])
        else:
            donch_high[i] = np.max(high_1d[i-19:i+1])
            donch_low[i] = np.min(low_1d[i-19:i+1])
    
    # === 1d Volume Confirmation ===
    vol_1d = df_1d['volume'].values
    vol_ma_20 = np.full_like(vol_1d, np.nan)
    for i in range(len(vol_ma_20)):
        if i < 19:
            if i == 0:
                vol_ma_20[i] = vol_1d[0]
            else:
                vol_ma_20[i] = (vol_ma_20[i-1] * i + vol_1d[i]) / (i + 1)
        else:
            vol_ma_20[i] = (vol_ma_20[i-1] * 19 + vol_1d[i]) / 20
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_confirm = vol_1d > vol_ma_20 * 1.5
    
    # === Align indicators to 12h timeframe ===
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1w, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1w, lower_keltner)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_keltner_aligned[i]) or 
            np.isnan(lower_keltner_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_confirm_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Market regime: Only trade when price is ABOVE weekly Keltner middle (uptrend bias)
        # For bear markets, we'll use the opposite - price BELOW Keltner middle
        # But to keep it simple and work in both regimes, we use breakout direction
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Price breaks above 1d Donchian High + weekly uptrend (price > weekly middle) + volume confirmation
            if (close[i] > donch_high_aligned[i] and 
                close[i] > (upper_keltner_aligned[i] + lower_keltner_aligned[i]) / 2 and  # price > weekly Keltner middle
                vol_confirm_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price breaks below 1d Donchian Low + weekly downtrend (price < weekly middle) + volume confirmation
            elif (close[i] < donch_low_aligned[i] and 
                  close[i] < (upper_keltner_aligned[i] + lower_keltner_aligned[i]) / 2 and  # price < weekly Keltner middle
                  vol_confirm_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal or volatility-based exit
        elif position == 1:
            # Exit long: Price breaks below 1d Donchian Low OR weekly downtrend
            if (close[i] < donch_low_aligned[i] or 
                close[i] < (upper_keltner_aligned[i] + lower_keltner_aligned[i]) / 2):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above 1d Donchian High OR weekly uptrend
            if (close[i] > donch_high_aligned[i] or 
                close[i] > (upper_keltner_aligned[i] + lower_keltner_aligned[i]) / 2):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Keltner_WeeklyTrend_DonchianBreakout_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0