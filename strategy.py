#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 level AND close > EMA34(1d) AND volume > 1.5x 20-bar average.
# Short when price breaks below Camarilla S3 level AND close < EMA34(1d) AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Camarilla levels provide precise intraday support/resistance from prior day's range.
# EMA34(1d) ensures alignment with higher timeframe trend, reducing counter-trend trades.
# Volume confirmation filters breakouts in low-participation markets.
# Primary timeframe: 4h, HTF: 1d for Camarilla levels and trend filter.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla levels and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous completed day
    # Camarilla R3 = close + 1.1*(high - low)/2
    # Camarilla S3 = close - 1.1*(high - low)/2
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 4h timeframe (available after 1d bar closes)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate EMA34(1d) for trend filter
    ema_34 = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: current 4h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume spike threshold
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_r3_aligned[i]  # break above R3
        breakout_down = curr_low < camarilla_s3_aligned[i]  # break below S3
        
        # Trend filter: close vs EMA34(1d)
        uptrend = curr_close > ema_34_aligned[i]
        downtrend = curr_close < ema_34_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3 AND uptrend AND volume confirmation
            if (breakout_up and 
                uptrend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3 AND downtrend AND volume confirmation
            elif (breakout_down and 
                  downtrend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S3 (stoploss) OR trend turns down
            if (curr_low < camarilla_s3_aligned[i] or 
                downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (stoploss) OR trend turns up
            if (curr_high > camarilla_r3_aligned[i] or 
                uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals