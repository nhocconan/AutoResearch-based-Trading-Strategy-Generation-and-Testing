#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter and volume spike.
# Long when price breaks above Camarilla R3 AND 1d close > 1d EMA34 AND volume > 2.0x 20-bar average.
# Short when price breaks below Camarilla S3 AND 1d close < 1d EMA34 AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Camarilla levels from 1d provide institutional structure; EMA34 filter ensures alignment with daily trend.
# Volume spike avoids false breakouts in low-volatility chop. Weekly timeframe HTF not needed as 1d suffices for 6h.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous completed 1d bar
    # Camarilla: based on (high, low, close) of previous day
    ph = df_1d['high'].shift(1).values  # previous day high
    pl = df_1d['low'].shift(1).values   # previous day low
    pc = df_1d['close'].shift(1).values # previous day close
    
    # Camarilla R3 and S3
    camarilla_r3 = pc + (ph - pl) * 1.1 / 4
    camarilla_s3 = pc - (ph - pl) * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: current 6h volume > 2.0x 20-bar average
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        
        # Camarilla breakout signals
        breakout_up = curr_high > camarilla_r3_aligned[i]  # break above R3
        breakout_down = curr_low < camarilla_s3_aligned[i]  # break below S3
        
        # 1d trend filter
        uptrend = close[i] > ema_34_aligned[i]   # current 6h close > aligned 1d EMA34
        downtrend = close[i] < ema_34_aligned[i] # current 6h close < aligned 1d EMA34
        
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