#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1-week volume confirmation and weekly trend filter.
Long when price breaks above R3 with weekly volume > 1.5x 10-week average and weekly close > weekly EMA(20);
Short when price breaks below S3 with same conditions reversed.
Exit on opposite Camarilla break or 1.5x ATR stop.
Designed for 15-25 trades/year to minimize fee drift while capturing institutional levels.
Works in bull via breakouts and bear via breakdowns with institutional volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_ata(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly indicators for trend and volume
    weekly_close = df_1w['close'].values
    weekly_volume = df_1w['volume'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Weekly EMA(20) for trend filter
    weekly_ema_20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly volume average (10-period)
    weekly_vol_avg = pd.Series(weekly_volume).rolling(window=10, min_periods=10).mean().values
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ATR for stop (daily)
    tr1 = df_1d['high'].values - df_1d['low'].values
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily = pd.Series(tr).rolling(window=5, min_periods=5).mean().values
    
    # Align weekly indicators to 12h timeframe
    weekly_ema_20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_20)
    weekly_vol_avg_aligned = align_htf_to_ltf(prices, df_1w, weekly_vol_avg)
    weekly_volume_aligned = align_htf_to_ltf(prices, df_1w, weekly_volume)
    atr_daily_aligned = align_htf_to_ltf(prices, df_1d, atr_daily)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(weekly_ema_20_aligned[i]) or np.isnan(weekly_vol_avg_aligned[i]) or
            np.isnan(weekly_volume_aligned[i]) or np.isnan(atr_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current day's OHLC for Camarilla calculation
        if i < len(df_1d):
            day_idx = i // 2  # Approximate: 2x 12h bars per day
            if day_idx >= len(df_1d):
                day_idx = len(df_1d) - 1
            if day_idx < 0:
                continue
                
            prev_high = df_1d['high'].iloc[day_idx-1] if day_idx > 0 else df_1d['high'].iloc[0]
            prev_low = df_1d['low'].iloc[day_idx-1] if day_idx > 0 else df_1d['low'].iloc[0]
            prev_close = df_1d['close'].iloc[day_idx-1] if day_idx > 0 else df_1d['close'].iloc[0]
            
            # Calculate Camarilla levels
            range_val = prev_high - prev_low
            if range_val <= 0:
                continue
                
            r3 = prev_close + (range_val * 1.1 / 4)
            s3 = prev_close - (range_val * 1.1 / 4)
            
            price_close = prices['close'].iloc[i]
            price_high = prices['high'].iloc[i]
            price_low = prices['low'].iloc[i]
            
            # Current values
            weekly_ema = weekly_ema_20_aligned[i]
            weekly_vol_avg_val = weekly_vol_avg_aligned[i]
            weekly_vol_current = weekly_volume_aligned[i]
            atr_val = atr_daily_aligned[i]
            
            if position == 0:
                # Enter long: break above R3 with volume surge and weekly close > weekly EMA20
                if (price_high > r3 and 
                    weekly_vol_current > 1.5 * weekly_vol_avg_val and
                    weekly_close[min(len(weekly_close)-1, i//28)] > weekly_ema):  # Approximate weekly index
                    signals[i] = 0.25
                    position = 1
                # Enter short: break below S3 with volume surge and weekly close < weekly EMA20
                elif (price_low < s3 and 
                      weekly_vol_current > 1.5 * weekly_vol_avg_val and
                      weekly_close[min(len(weekly_close)-1, i//28)] < weekly_ema):
                    signals[i] = -0.25
                    position = -1
            
            elif position != 0:
                # Exit: opposite Camarilla break or 1.5x ATR stop
                exit_signal = False
                
                if position == 1:
                    # Exit long: break below S3 OR price < entry - 1.5*ATR
                    if price_low < s3:
                        exit_signal = True
                    else:
                        entry_level = r3  # Approximate entry at R3 break
                        if price_close < entry_level - 1.5 * atr_val:
                            exit_signal = True
                elif position == -1:
                    # Exit short: break above R3 OR price > entry + 1.5*ATR
                    if price_high > r3:
                        exit_signal = True
                    else:
                        entry_level = s3  # Approximate entry at S3 break
                        if price_close > entry_level + 1.5 * atr_val:
                            exit_signal = True
                
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    # Hold position
                    signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R3S3_VolumeSurge1.5x_WeeklyEMA20Trend_ATR1.5x"
timeframe = "12h"
leverage = 1.0