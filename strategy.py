#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter (EMA34) and volume spike
# Long when price breaks above Camarilla R3 AND close > EMA34(1d) AND volume > 2.0x 20-period average
# Short when price breaks below Camarilla S3 AND close < EMA34(1d) AND volume > 2.0x 20-period average
# Exit when price retracement to Camarilla pivot point (PP) OR EMA34(1d) trend flip
# Uses 12h primary timeframe with 1d HTF for trend filter to reduce whipsaw
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Camarilla levels provide precise intraday support/resistance; volume confirmation filters false breaks;
# 1d EMA avoids counter-trend trades in bear markets

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
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
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels on 12h data (using previous bar's OHLC)
    # Camarilla: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_pp = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    if len(high) >= 2:
        for i in range(1, n):
            camarilla_pp[i] = (high[i-1] + low[i-1] + close[i-1]) / 3.0
            camarilla_r3[i] = close[i-1] + (high[i-1] - low[i-1]) * 1.1 / 2.0
            camarilla_s3[i] = close[i-1] - (high[i-1] - low[i-1]) * 1.1 / 2.0
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_pp[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND close > EMA34(1d) AND volume spike
            if (high[i] > camarilla_r3[i] and  # breakout above R3 level
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND close < EMA34(1d) AND volume spike
            elif (low[i] < camarilla_s3[i] and  # breakout below S3 level
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retracement to Camarilla PP OR close < EMA34(1d) (trend flip)
            if close[i] <= camarilla_pp[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retracement to Camarilla PP OR close > EMA34(1d) (trend flip)
            if close[i] >= camarilla_pp[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals