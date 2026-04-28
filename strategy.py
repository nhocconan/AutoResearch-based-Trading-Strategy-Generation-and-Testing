#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price > R3 AND close > 1d EMA34 AND volume > 1.5x 20-bar avg
# Short when price < S3 AND close < 1d EMA34 AND volume > 1.5x 20-bar avg
# Exits when price crosses back through the central pivot (PP) or volume drops
# Target: 12-37 trades/year via tight entry conditions and trend filter
# Camarilla levels provide structured support/resistance, EMA34 filters trend direction,
# volume confirmation avoids false breakouts. Works in both bull and bear by only
# trading in direction of 1d trend with clear structure.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_TrendFilter_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 and Camarilla calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pp = (high_1d + low_1d + close_1d) / 3.0
    r3 = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    s3 = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align 1d indicators to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)  # for exit condition
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        price = close[i]
        ema_trend = ema_34_1d_aligned[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        pp_level = pp_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price > R3 AND close > 1d EMA34 (uptrend) AND volume confirmation
            if price > r3_level and price > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price < S3 AND close < 1d EMA34 (downtrend) AND volume confirmation
            elif price < s3_level and price < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price crosses below PP or volume drops
            if price < pp_level or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price crosses above PP or volume drops
            if price > pp_level or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals