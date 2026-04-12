#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    H_minus_L = prev_high - prev_low
    R4 = prev_close + H_minus_L * 1.1 / 2
    R3 = prev_close + H_minus_L * 1.1 / 4
    S3 = prev_close - H_minus_L * 1.1 / 4
    S4 = prev_close - H_minus_L * 1.1 / 2
    
    # Map daily Camarilla levels to each 12h bar
    R4_mapped = np.full(n, np.nan)
    R3_mapped = np.full(n, np.nan)
    S3_mapped = np.full(n, np.nan)
    S4_mapped = np.full(n, np.nan)
    
    for i in range(n):
        current_time = pd.Timestamp(prices.iloc[i]['open_time'])
        # Find the day that ended before this 12h bar
        for j in range(len(df_1d)):
            day_start = pd.Timestamp(df_1d.iloc[j]['open_time'])
            day_end = day_start + pd.Timedelta(days=1)
            if day_start <= current_time < day_end:
                R4_mapped[i] = R4[j]
                R3_mapped[i] = R3[j]
                S3_mapped[i] = S3[j]
                S4_mapped[i] = S4[j]
                break
    
    # Daily ATR for volatility filter
    tr1 = df_1d['high'].values[1:] - df_1d['low'].values[1:]
    tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr_daily = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_daily = pd.Series(tr_daily).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Map ATR to 12h timeframe
    atr_daily_mapped = np.full(n, np.nan)
    for i in range(n):
        current_time = pd.Timestamp(prices.iloc[i]['open_time'])
        for j in range(len(df_1d)):
            day_start = pd.Timestamp(df_1d.iloc[j]['open_time'])
            day_end = day_start + pd.Timedelta(days=1)
            if day_start <= current_time < day_end:
                atr_daily_mapped[i] = atr_daily[j]
                break
    
    # Volume confirmation: current 12h volume > 20-period average of daily volume
    vol_1d_mapped = np.full(n, np.nan)
    for i in range(n):
        current_time = pd.Timestamp(prices.iloc[i]['open_time'])
        for j in range(len(df_1d)):
            day_start = pd.Timestamp(df_1d.iloc[j]['open_time'])
            day_end = day_start + pd.Timedelta(days=1)
            if day_start <= current_time < day_end:
                vol_1d_mapped[i] = df_1d.iloc[j]['volume']
                break
    vol_ma = pd.Series(vol_1d_mapped).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    # Chop index for regime filter (12h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    atr_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    chop_raw = 100 * np.log10(tr_sum / (atr_12h * 14)) / np.log10(14)
    chop = np.where(tr_sum > 0, chop_raw, 50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(R4_mapped[i]) or np.isnan(R3_mapped[i]) or 
            np.isnan(S3_mapped[i]) or np.isnan(S4_mapped[i]) or
            np.isnan(atr_daily_mapped[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility filter: only trade when volatility is elevated (ATR > 20-period average)
        vol_filter = atr_daily_mapped[i] > np.nanmedian(atr_daily_mapped[max(0, i-20):i+1])
        
        # Chop regime: Chop < 40 = trending (favor breakouts), Chop > 60 = ranging (avoid)
        trending_regime = chop[i] < 40
        
        # Long: price breaks above R4 (strong resistance) in trending market with volume
        long_signal = (close[i] > R4_mapped[i] and trending_regime and volume_filter[i] and vol_filter)
        
        # Short: price breaks below S4 (strong support) in trending market with volume
        short_signal = (close[i] < S4_mapped[i] and trending_regime and volume_filter[i] and vol_filter)
        
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