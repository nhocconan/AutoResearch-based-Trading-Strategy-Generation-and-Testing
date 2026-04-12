#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_volume_v1"
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
    
    # 12h data for Camarilla levels (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate previous 12h bar's Camarilla levels
    prev_close = df_12h['close'].shift(1).values
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    
    H_minus_L = prev_high - prev_low
    R4 = prev_close + H_minus_L * 1.1 / 2
    R3 = prev_close + H_minus_L * 1.1 / 4
    S3 = prev_close - H_minus_L * 1.1 / 4
    S4 = prev_close - H_minus_L * 1.1 / 2
    
    # Map 12h Camarilla levels to each 4h bar
    R4_mapped = np.full(n, np.nan)
    R3_mapped = np.full(n, np.nan)
    S3_mapped = np.full(n, np.nan)
    S4_mapped = np.full(n, np.nan)
    
    for i in range(n):
        current_time = pd.Timestamp(prices.iloc[i]['open_time'])
        # Find the 12h bar that ended before this 4h bar
        for j in range(len(df_12h)):
            bar_start = pd.Timestamp(df_12h.iloc[j]['open_time'])
            bar_end = bar_start + pd.Timedelta(hours=12)
            if bar_start <= current_time < bar_end:
                R4_mapped[i] = R4[j]
                R3_mapped[i] = R3[j]
                S3_mapped[i] = S3[j]
                S4_mapped[i] = S4[j]
                break
    
    # Volume confirmation: current 4h volume > 20-period average of 12h volume
    vol_12h_mapped = np.full(n, np.nan)
    for i in range(n):
        current_time = pd.Timestamp(prices.iloc[i]['open_time'])
        for j in range(len(df_12h)):
            bar_start = pd.Timestamp(df_12h.iloc[j]['open_time'])
            bar_end = bar_start + pd.Timedelta(hours=12)
            if bar_start <= current_time < bar_end:
                vol_12h_mapped[i] = df_12h.iloc[j]['volume']
                break
    vol_ma = pd.Series(vol_12h_mapped).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    # Chop index for regime filter (4h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    atr_4h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    chop_raw = 100 * np.log10(tr_sum / (atr_4h * 14)) / np.log10(14)
    chop = np.where(tr_sum > 0, chop_raw, 50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(R4_mapped[i]) or np.isnan(R3_mapped[i]) or 
            np.isnan(S3_mapped[i]) or np.isnan(S4_mapped[i]) or
            np.isnan(chop[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Chop regime: Chop < 40 = trending (favor breakouts), Chop > 60 = ranging (avoid)
        trending_regime = chop[i] < 40
        
        # Long: price breaks above R4 (strong resistance) in trending market with volume
        long_signal = (close[i] > R4_mapped[i] and trending_regime and volume_filter[i])
        
        # Short: price breaks below S4 (strong support) in trending market with volume
        short_signal = (close[i] < S4_mapped[i] and trending_regime and volume_filter[i])
        
        # Exit: chop increases (range) or price returns to mid-point (S3/R3)
        exit_long = (position == 1 and (chop[i] > 60 or close[i] < (R3_mapped[i] + S3_mapped[i])/2))
        exit_short = (position == -1 and (chop[i] > 60 or close[i] > (R3_mapped[i] + S3_mapped[i])/2))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals