#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 Breakout with 1d Volume Spike and ADX Regime Filter
# Long when price breaks above Camarilla R3 AND 1d volume > 2.0x 20-period average AND 1d ADX > 25 (trending)
# Short when price breaks below Camarilla S3 AND 1d volume > 2.0x 20-period average AND 1d ADX > 25 (trending)
# Exit when price crosses 1d EMA34 (trend reversal) OR opposite Camarilla level is touched
# Uses 12h primary timeframe with 1d HTF for volume, ADX, and EMA34 filters to capture sustained moves with low frequency
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Combines Camarilla precision with volume confirmation and ADX trend strength to avoid whipsaws

name = "12h_Camarilla_R3S3_Breakout_1dVolume_ADX_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend exit
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ADX on 1d for trend filter (ADX > 25 = trending)
    if len(df_1d) >= 14:
        # True Range
        tr1 = pd.Series(df_1d['high']).diff().abs()
        tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close'].shift())).abs()
        tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_1d = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
        
        # Directional Movement
        up_move = pd.Series(df_1d['high']).diff()
        down_move = -pd.Series(df_1d['low']).diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed DM
        plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
        minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    else:
        adx_1d_aligned = np.zeros(n)
    
    # Calculate 1d volume MA for volume spike filter
    if len(df_1d) >= 20:
        vol_ma_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
        vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
        volume_filter = df_1d['volume'].values > (2.0 * vol_ma_20_1d)
        volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter.astype(float))
    else:
        volume_filter_aligned = np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h bar (use shift to avoid look-ahead)
    prev_high = np.concatenate([[high[0]], high[:-1]])  # shift(1)
    prev_low = np.concatenate([[low[0]], low[:-1]])
    prev_close = np.concatenate([[close[0]], close[:-1]])
    
    rang = prev_high - prev_low
    camarilla_r3 = prev_close + rang * 1.1 / 4.0
    camarilla_s3 = prev_close - rang * 1.1 / 4.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 1d ADX > 25 AND 1d volume spike
            if (close[i] > camarilla_r3[i] and 
                adx_1d_aligned[i] > 25 and 
                volume_filter_aligned[i] > 0.5):  # aligned volume filter is 0.0 or 1.0
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 1d ADX > 25 AND 1d volume spike
            elif (close[i] < camarilla_s3[i] and 
                  adx_1d_aligned[i] > 25 and 
                  volume_filter_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA34 (trend reversal) OR touches Camarilla S3 (support)
            if close[i] < ema_34_1d_aligned[i] or close[i] <= camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d EMA34 (trend reversal) OR touches Camarilla R3 (resistance)
            if close[i] > ema_34_1d_aligned[i] or close[i] >= camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals