#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA34 + 1w EMA20 with volume confirmation
# Long when Alligator jaws (13-period SMA of median price) < teeth (8-period SMA of median price) < lips (5-period SMA of median price) and price > 1d EMA34 and price > 1w EMA20
# Short when Alligator jaws > teeth > lips and price < 1d EMA34 and price < 1w EMA20
# Exit when Alligator lines cross in opposite direction or price crosses opposite EMA
# Uses Alligator for trend direction and alignment, EMA for multi-timeframe confirmation, volume for filtering
# Targets 15-25 trades/year to minimize fee drag while capturing strong trends

name = "12h_Williams_Alligator_1dEMA34_1wEMA20_Volume"
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
    
    # Get weekly data for EMA20
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Get daily data for EMA34 and volume
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA20
    weekly_close = df_weekly['close'].values
    ema_20w = np.full_like(weekly_close, np.nan, dtype=float)
    if len(weekly_close) >= 20:
        ema_20w[19] = np.mean(weekly_close[:20])
        for i in range(20, len(weekly_close)):
            ema_20w[i] = (weekly_close[i] * 2 + ema_20w[i-1] * 18) / 20
    
    # Calculate daily EMA34
    daily_close = df_daily['close'].values
    ema_34d = np.full_like(daily_close, np.nan, dtype=float)
    if len(daily_close) >= 34:
        ema_34d[33] = np.mean(daily_close[:34])
        for i in range(34, len(daily_close)):
            ema_34d[i] = (daily_close[i] * 2 + ema_34d[i-1] * 32) / 34
    
    # Calculate daily average volume for volume filter
    daily_volume = df_daily['volume'].values
    vol_ma_30 = np.full_like(daily_volume, np.nan, dtype=float)
    if len(daily_volume) >= 30:
        vol_ma_30[29] = np.mean(daily_volume[:30])
        for i in range(30, len(daily_volume)):
            vol_ma_30[i] = (daily_volume[i] * 2 + vol_ma_30[i-1] * 28) / 30
    
    # Calculate Williams Alligator on 12h data (median price)
    median_price = (high + low) / 2
    
    # Jaws: 13-period SMMA of median price
    jaws = np.full(n, np.nan, dtype=float)
    if n >= 13:
        jaws[12] = np.mean(median_price[:13])
        for i in range(13, n):
            jaws[i] = (median_price[i] + jaws[i-1] * 12) / 13
    
    # Teeth: 8-period SMMA of median price
    teeth = np.full(n, np.nan, dtype=float)
    if n >= 8:
        teeth[7] = np.mean(median_price[:8])
        for i in range(8, n):
            teeth[i] = (median_price[i] + teeth[i-1] * 7) / 8
    
    # Lips: 5-period SMMA of median price
    lips = np.full(n, np.nan, dtype=float)
    if n >= 5:
        lips[4] = np.mean(median_price[:5])
        for i in range(5, n):
            lips[i] = (median_price[i] + lips[i-1] * 4) / 5
    
    # Align weekly EMA20 to 12h timeframe
    ema_20w_aligned = align_htf_to_ltf(prices, df_weekly, ema_20w)
    
    # Align daily EMA34 to 12h timeframe
    ema_34d_aligned = align_htf_to_ltf(prices, df_daily, ema_34d)
    
    # Align daily volume MA to 12h timeframe
    vol_ma_30_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_20w_aligned[i]) or np.isnan(ema_34d_aligned[i]) or 
            np.isnan(vol_ma_30_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume > 1.3x 30-day SMA
        vol_filter = volume[i] > 1.3 * vol_ma_30_aligned[i]
        
        if position == 0:
            # Look for Alligator alignment with EMA confirmation and volume
            # Long: jaws < teeth < lips and price > both EMAs
            if jaws[i] < teeth[i] < lips[i] and close[i] > ema_34d_aligned[i] and close[i] > ema_20w_aligned[i]:
                if vol_filter:
                    signals[i] = 0.25
                    position = 1
            # Short: jaws > teeth > lips and price < both EMAs
            elif jaws[i] > teeth[i] > lips[i] and close[i] < ema_34d_aligned[i] and close[i] < ema_20w_aligned[i]:
                if vol_filter:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Alligator reverses (jaws > teeth) or price crosses below either EMA
            if jaws[i] > teeth[i] or close[i] < ema_34d_aligned[i] or close[i] < ema_20w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator reverses (jaws < teeth) or price crosses above either EMA
            if jaws[i] < teeth[i] or close[i] > ema_34d_aligned[i] or close[i] > ema_20w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals