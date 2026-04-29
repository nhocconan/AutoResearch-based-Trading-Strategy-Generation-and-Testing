#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation (>1.5x 20-period average)
# Camarilla levels identify intraday support/resistance where reversals/breakouts occur; R3/S3 are strong breakout levels.
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws in both bull/bear markets.
# Volume confirmation filters for institutional participation; discrete sizing (0.25) minimizes fee churn.
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for each 4h bar using previous day's OHLC
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_mid = np.full(n, np.nan)
    
    for i in range(n):
        # Get previous completed 1d bar for Camarilla calculation
        # Find the 1d bar that closed before current 4h bar's open time
        curr_time = prices.iloc[i]['open_time']
        # Get all 1d bars that closed before current time
        mask = df_1d['open_time'] < curr_time
        if not mask.any():
            continue
        prev_1d_idx = mask.sum() - 1  # index of last completed 1d bar
        if prev_1d_idx < 0:
            continue
            
        # Previous day's OHLC
        prev_high = df_1d.iloc[prev_1d_idx]['high']
        prev_low = df_1d.iloc[prev_1d_idx]['low']
        prev_close = df_1d.iloc[prev_1d_idx]['close']
        
        # Calculate Camarilla levels
        range_hl = prev_high - prev_low
        camarilla_r3[i] = prev_close + range_hl * 1.1 / 4
        camarilla_s3[i] = prev_close - range_hl * 1.1 / 4
        camarilla_mid[i] = (camarilla_r3[i] + camarilla_s3[i]) / 2
    
    # Calculate 20-period average volume for confirmation (on 4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # 1d EMA34, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(camarilla_mid[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        curr_mid = camarilla_mid[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Camarilla breakout conditions
        breakout_long = curr_high > curr_r3   # price breaks above R3
        breakout_short = curr_low < curr_s3   # price breaks below S3
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price retouches midpoint OR trend turns bearish
            if curr_close < curr_mid or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retouches midpoint OR trend turns bullish
            if curr_close > curr_mid or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: bullish breakout AND above 1d EMA34 AND volume confirmation
            if (breakout_long and 
                curr_close > curr_ema_1d and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish breakout AND below 1d EMA34 AND volume confirmation
            elif (breakout_short and 
                  curr_close < curr_ema_1d and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals