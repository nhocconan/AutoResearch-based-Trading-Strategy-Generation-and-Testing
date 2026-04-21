#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirm_v1
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h EMA50 trend filter and volume spike confirmation. Designed for 1h timeframe with tight entries (15-37/year) to minimize fee drag. Uses 4h HTF for trend and volume context, 1h only for precise entry timing. Session filter (08-20 UTC) reduces noise. Position size 0.20.
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
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # === 4h trend filter: 50-period EMA ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 4h volume average (20-period) for spike detection ===
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h[np.isnan(vol_ma_4h)] = 1.0  # avoid division by zero
    vol_ratio_4h = volume_4h / vol_ma_4h
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    # === Calculate Camarilla levels from previous day on 1h ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Group by date to get daily OHLC for Camarilla calculation
    dates = pd.to_datetime(prices['open_time']).date
    unique_dates = np.unique(dates)
    
    # Arrays to store Camarilla levels for each 1h bar
    camarilla_R1 = np.full(n, np.nan)
    camarilla_S1 = np.full(n, np.nan)
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    camarilla_close = np.full(n, np.nan)
    
    for date in unique_dates:
        mask = (dates == date)
        if not np.any(mask):
            continue
        
        # Get prior day's OHLC (shift by 1)
        prev_date_idx = np.where(unique_dates == date)[0][0] - 1
        if prev_date_idx < 0:
            continue
        prev_date = unique_dates[prev_date_idx]
        prev_mask = (dates == prev_date)
        
        if not np.any(prev_mask):
            continue
            
        # Prior day's OHLC
        ph = high[prev_mask].max()
        pl = low[prev_mask].min()
        pc = close[prev_mask][-1]  # last close of prior day
        
        # Camarilla levels
        range_ = ph - pl
        camarilla_close_val = pc
        camarilla_R1_val = pc + (range_ * 1.1 / 12)
        camarilla_S1_val = pc - (range_ * 1.1 / 12)
        camarilla_R3_val = pc + (range_ * 1.1 / 4)
        camarilla_S3_val = pc - (range_ * 1.1 / 4)
        
        # Assign to today's bars
        camarilla_close[mask] = camarilla_close_val
        camarilla_R1[mask] = camarilla_R1_val
        camarilla_S1[mask] = camarilla_S1_val
        camarilla_R3[mask] = camarilla_R3_val
        camarilla_S3[mask] = camarilla_S3_val
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.to_datetime(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_ratio_4h_aligned[i]) or
            np.isnan(camarilla_R1[i]) or np.isnan(camarilla_S1[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        trend_4h = ema_50_4h_aligned[i]
        vol_spike = vol_ratio_4h_aligned[i]
        R1 = camarilla_R1[i]
        S1 = camarilla_S1[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume spike > 2.0 + price above 4h EMA50
            if price_close > R1 and vol_spike > 2.0 and price_close > trend_4h:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + volume spike > 2.0 + price below 4h EMA50
            elif price_close < S1 and vol_spike > 2.0 and price_close < trend_4h:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions: reverse signal or loss of trend/volume
            if position == 1:
                # Exit long: price below S1 or loss of trend or low volume
                if price_close < S1 or price_close < trend_4h or vol_spike < 1.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                # Exit short: price above R1 or loss of trend or low volume
                if price_close > R1 or price_close > trend_4h or vol_spike < 1.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0