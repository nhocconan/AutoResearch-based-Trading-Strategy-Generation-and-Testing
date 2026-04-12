#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_keltner_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Keltner Channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Map daily data to each 12h bar (previous day's values)
    high_1d_mapped = np.full(n, np.nan)
    low_1d_mapped = np.full(n, np.nan)
    close_1d_mapped = np.full(n, np.nan)
    
    for i in range(n):
        current_time = pd.Timestamp(prices.iloc[i]['open_time'])
        prev_date = current_time.date() - pd.Timedelta(days=1)
        
        # Find previous day in daily data
        for j in range(len(df_1d)):
            if pd.Timestamp(df_1d.iloc[j]['open_time']).date() == prev_date:
                high_1d_mapped[i] = high_1d[j]
                low_1d_mapped[i] = low_1d[j]
                close_1d_mapped[i] = close_1d[j]
                break
    
    # Calculate ATR (10-period) on daily data
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_daily = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_daily = pd.Series(tr_daily).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channel: 20 EMA ± 2 * ATR
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_20 + 2 * atr_daily
    lower_keltner = ema_20 - 2 * atr_daily
    
    # Map Keltner levels to 12h timeframe
    upper_keltner_mapped = np.full(n, np.nan)
    lower_keltner_mapped = np.full(n, np.nan)
    ema_20_mapped = np.full(n, np.nan)
    
    for i in range(n):
        current_time = pd.Timestamp(prices.iloc[i]['open_time'])
        prev_date = current_time.date() - pd.Timedelta(days=1)
        
        for j in range(len(df_1d)):
            if pd.Timestamp(df_1d.iloc[j]['open_time']).date() == prev_date:
                upper_keltner_mapped[i] = upper_keltner[j]
                lower_keltner_mapped[i] = lower_keltner[j]
                ema_20_mapped[i] = ema_20[j]
                break
    
    # Volume confirmation: current 12h volume > 20-period average of mapped daily volume
    vol_1d_mapped = np.full(n, np.nan)
    for i in range(n):
        current_time = pd.Timestamp(prices.iloc[i]['open_time'])
        prev_date = current_time.date() - pd.Timedelta(days=1)
        for j in range(len(df_1d)):
            if pd.Timestamp(df_1d.iloc[j]['open_time']).date() == prev_date:
                vol_1d_mapped[i] = df_1d.iloc[j]['volume']
                break
    vol_ma = pd.Series(vol_1d_mapped).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    # Chop index for regime filter (12h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop_raw = 100 * np.log10(tr_sum / (atr_12h * 14)) / np.log10(14)
    chop = np.where(tr_sum > 0, chop_raw, 50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if not ready
        if (np.isnan(upper_keltner_mapped[i]) or np.isnan(lower_keltner_mapped[i]) or
            np.isnan(ema_20_mapped[i]) or np.isnan(chop[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below EMA20
        uptrend = close[i] > ema_20_mapped[i]
        downtrend = close[i] < ema_20_mapped[i]
        
        # Chop regime: Chop < 40 = trending (favor breakouts), Chop > 60 = ranging (avoid)
        trending_regime = chop[i] < 40
        
        # Long: price breaks above upper Keltner in uptrend + volume
        long_signal = (close[i] > upper_keltner_mapped[i] and uptrend and trending_regime and volume_filter[i])
        
        # Short: price breaks below lower Keltner in downtrend + volume
        short_signal = (close[i] < lower_keltner_mapped[i] and downtrend and trending_regime and volume_filter[i])
        
        # Exit: chop increases (range) or price returns to EMA20
        exit_long = (position == 1 and (chop[i] > 60 or close[i] < ema_20_mapped[i]))
        exit_short = (position == -1 and (chop[i] > 60 or close[i] > ema_20_mapped[i]))
        
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