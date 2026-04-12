#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for ATR and volume
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly ATR for volatility filter
    tr1 = df_1w['high'].values[1:] - df_1w['low'].values[1:]
    tr2 = np.abs(df_1w['high'].values[1:] - df_1w['close'].values[:-1])
    tr3 = np.abs(df_1w['low'].values[1:] - df_1w['close'].values[:-1])
    tr_weekly = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_weekly_raw = pd.Series(tr_weekly).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Map ATR to daily timeframe using proper alignment
    atr_weekly = align_htf_to_ltf(prices, df_1w, atr_weekly_raw)
    
    # Weekly volume average for volume confirmation
    vol_weekly = df_1w['volume'].values
    vol_weekly_avg = pd.Series(vol_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_weekly_avg_aligned = align_htf_to_ltf(prices, df_1w, vol_weekly_avg)
    
    # Daily Chop index for regime filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_daily = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop_raw = 100 * np.log10(tr_sum / (atr_daily * 14)) / np.log10(14)
    chop = np.where(tr_sum > 0, chop_raw, 50)
    
    # Calculate daily pivot points for entry levels
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if not ready
        if np.isnan(atr_weekly[i]) or np.isnan(vol_weekly_avg_aligned[i]) or np.isnan(chop[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility filter: only trade when volatility is elevated
        vol_filter = atr_weekly[i] > np.nanmedian(atr_weekly[max(0, i-20):i+1])
        
        # Chop regime: Chop < 40 = trending (favor breakouts), Chop > 60 = ranging (avoid)
        trending_regime = chop[i] < 40
        
        # Long: price breaks above R1 in trending market with volume
        long_signal = (close[i] > r1[i] and trending_regime and volume[i] > vol_weekly_avg_aligned[i] and vol_filter)
        
        # Short: price breaks below S1 in trending market with volume
        short_signal = (close[i] < s1[i] and trending_regime and volume[i] > vol_weekly_avg_aligned[i] and vol_filter)
        
        # Exit: chop increases (range) or price returns to pivot
        exit_long = (position == 1 and (chop[i] > 60 or close[i] < pivot[i]))
        exit_short = (position == -1 and (chop[i] > 60 or close[i] > pivot[i]))
        
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