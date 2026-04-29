#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume spike
# Long when price breaks above Camarilla R3 AND price > 1w EMA34 AND volume > 2.0x 20-period average
# Short when price breaks below Camarilla S3 AND price < 1w EMA34 AND volume > 2.0x 20-period average
# Uses discrete position sizing (0.25) to minimize fee drag
# Camarilla levels from daily timeframe provide strong intraday support/resistance
# Weekly EMA filter ensures we trade with the higher timeframe trend
# Volume confirmation adds momentum validity to breakouts
# Designed for low trade frequency (target: 12-37/year) to avoid fee drag

name = "12h_Camarilla_R3S3_Breakout_1wEMA34_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 40 or len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from daily timeframe
    # Camarilla: based on previous day's OHLC
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Typical price for Camarilla calculation
    typical_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3, R2, R1, PP, S1, S2, S3
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    camarilla_r3 = close_1d + (range_1d * 1.1 / 2.0)
    camarilla_s3 = close_1d - (range_1d * 1.1 / 2.0)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate ATR for volatility (optional, can be used for position sizing)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = np.zeros(n)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_ema_1w = ema_34_1w_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_vol_spike = vol_spike[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price breaks below S3 or loses weekly EMA trend
            if curr_close < curr_s3 or curr_close < curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 or gains above weekly EMA (for short)
            if curr_close > curr_r3 or curr_close > curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND price > 1w EMA34 AND volume spike
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_1w and 
                curr_vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 AND price < 1w EMA34 AND volume spike
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_1w and 
                  curr_vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals