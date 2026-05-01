#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with weekly pivot direction filter and volume confirmation
# Uses weekly Camarilla pivot levels to establish bias (long above weekly R3, short below weekly S3)
# 6h Camarilla R3/S3 breakout in direction of weekly bias with volume spike (2.0x 20-period MA)
# Works in bull/bear via weekly pivot filter - avoids counter-trend trades
# Designed for low frequency (50-150 trades over 4 years) to minimize fee drag on 6h timeframe

name = "6h_Camarilla_R3S3_Breakout_WeeklyPivot_Dir_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for weekly pivot bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Weekly Camarilla levels (using prior weekly bar's HLC)
    prior_weekly_high = np.concatenate([[np.nan], df_1w['high'].values[:-1]])
    prior_weekly_low = np.concatenate([[np.nan], df_1w['low'].values[:-1]])
    prior_weekly_close = np.concatenate([[np.nan], df_1w['close'].values[:-1]])
    
    weekly_hl_range = prior_weekly_high - prior_weekly_low
    weekly_r3 = prior_weekly_close + weekly_hl_range * 1.1 / 4
    weekly_s3 = prior_weekly_close - weekly_hl_range * 1.1 / 4
    
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3 - weekly_s3)  # dummy array for alignment
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    
    # 1d HTF data for prior day's HLC (used for 6h Camarilla calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1d HLC for 6h Camarilla levels
    prior_daily_high = np.concatenate([[np.nan], df_1d['high'].values[:-1]])
    prior_daily_low = np.concatenate([[np.nan], df_1d['low'].values[:-1]])
    prior_daily_close = np.concatenate([[np.nan], df_1d['close'].values[:-1]])
    
    daily_hl_range = prior_daily_high - prior_daily_low
    camarilla_r3 = prior_daily_close + daily_hl_range * 1.1 / 4
    camarilla_s3 = prior_daily_close - daily_hl_range * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(20, 5)  # Need volume MA20 and weekly data
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_r3_aligned[i]) or np.isnan(weekly_s3_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Weekly bias: price above weekly R3 = bullish bias, below weekly S3 = bearish bias
        bullish_bias = close[i] > weekly_r3_aligned[i]
        bearish_bias = close[i] < weekly_s3_aligned[i]
        
        # 6h Camarilla breakout conditions
        breakout_long = close[i] > camarilla_r3_aligned[i]  # Price breaks above R3
        breakout_short = close[i] < camarilla_s3_aligned[i]  # Price breaks below S3
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R3 with volume spike and bullish weekly bias
            if breakout_long and vol_spike and bullish_bias:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below S3 with volume spike and bearish weekly bias
            elif breakout_short and vol_spike and bearish_bias:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below prior 6h bar's low or weekly bias turns bearish
            prior_low = np.concatenate([[np.nan], low[:-1]])[i]
            if close[i] < prior_low or (close[i] < weekly_s3_aligned[i] and bearish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above prior 6h bar's high or weekly bias turns bullish
            prior_high = np.concatenate([[np.nan], high[:-1]])[i]
            if close[i] > prior_high or (close[i] > weekly_r3_aligned[i] and bullish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals