#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and choppiness regime filter.
# Long when price breaks above Camarilla R3 AND 1d volume > 2.0x 20-bar average AND CHOP(14) < 61.8 (trending regime).
# Short when price breaks below Camarilla S3 AND 1d volume > 2.0x 20-bar average AND CHOP(14) < 61.8.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Camarilla levels from 1d provide strong intraday support/resistance. Volume spike confirms breakout strength.
# Choppiness filter avoids ranging markets where breakouts fail. Weekly trend alignment reduces false signals.
# Primary timeframe: 12h, HTF: 1d for Camarilla levels and volume, 1w for trend bias.

name = "12h_Camarilla_R3S3_Breakout_1dVolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous completed 1d bar
    # Camarilla R3 = close + 1.1*(high - low)/2
    # Camarilla S3 = close - 1.1*(high - low)/2
    hl_range = df_1d['high'].values - df_1d['low'].values
    camarilla_r3 = df_1d['close'].values + 1.1 * hl_range / 2.0
    camarilla_s3 = df_1d['close'].values - 1.1 * hl_range / 2.0
    
    # Shift by 1 to use only completed 1d bar (avoid look-ahead)
    camarilla_r3 = np.roll(camarilla_r3, 1)
    camarilla_s3 = np.roll(camarilla_s3, 1)
    camarilla_r3[0] = np.nan  # First value invalid after roll
    camarilla_s3[0] = np.nan
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d volume confirmation: current 12h volume > 2.0x 20-bar average of 1d volume
    # We need to compare 12h volume against 1d volume average, so we use 1d volume data
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d = np.roll(vol_ma_1d, 1)  # Shift for completed bar
    vol_ma_1d[0] = np.nan
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Choppiness Index (CHOP) on 1d data for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (n * (HHH - LLL))) / log10(n)
    # Where ATR = TR, HHH = highest high, LLL = lowest low over period
    tr1 = np.maximum(df_1d['high'].values, np.roll(df_1d['close'].values, 1)) - np.minimum(df_1d['low'].values, np.roll(df_1d['close'].values, 1))
    tr1[0] = df_1d['high'].values[0] - df_1d['low'].values[0]  # First TR
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr14 * 14 / (highest_high - lowest_low)) / np.log10(14)
    chop_raw = np.roll(chop_raw, 1)  # Shift for completed bar
    chop_raw[0] = np.nan
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Weekly trend bias: 1 = bullish week (close > open), -1 = bearish week (close < open)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    weekly_bias_raw = np.where(df_1w['close'].values > df_1w['open'].values, 1,
                               np.where(df_1w['close'].values < df_1w['open'].values, -1, 0))
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_aligned[i]) or \
           np.isnan(weekly_bias_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        
        # Volume confirmation: current 12h volume > 2.0x 20-bar average of 1d volume
        if vol_ma_1d_aligned[i] <= 0:
            volume_confirm = False
        else:
            volume_confirm = curr_vol > (vol_ma_1d_aligned[i] * 2.0)
        
        # Choppiness regime filter: CHOP < 61.8 = trending regime (favor breakouts)
        trending_regime = chop_aligned[i] < 61.8
        
        # Breakout signals
        breakout_up = curr_high > camarilla_r3_aligned[i]  # break above Camarilla R3
        breakout_down = curr_low < camarilla_s3_aligned[i]  # break below Camarilla S3
        
        # Weekly bias filter
        bullish_week = weekly_bias_aligned[i] == 1
        bearish_week = weekly_bias_aligned[i] == -1
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3 AND volume confirmation AND trending regime AND bullish week
            if (breakout_up and 
                volume_confirm and 
                trending_regime and 
                bullish_week):
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3 AND volume confirmation AND trending regime AND bearish week
            elif (breakout_down and 
                  volume_confirm and 
                  trending_regime and 
                  bearish_week):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Camarilla S3 (stoploss) OR weekly bias turns bearish OR chop > 61.8 (ranging)
            if (curr_low < camarilla_s3_aligned[i] or 
                weekly_bias_aligned[i] == -1 or 
                chop_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Camarilla R3 (stoploss) OR weekly bias turns bullish OR chop > 61.8 (ranging)
            if (curr_high > camarilla_r3_aligned[i] or 
                weekly_bias_aligned[i] == 1 or 
                chop_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals