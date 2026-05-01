#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend and volume spike.
# Long when price breaks above Camarilla R3 AND price > EMA34(1d) AND volume > 2.0x 20-bar average.
# Short when price breaks below Camarilla S3 AND price < EMA34(1d) AND volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 75-200 total trades over 4 years (19-50/year).
# Camarilla levels provide precise intraday support/resistance, EMA34 filters trend direction,
# volume spike confirms breakout strength. Weekly HTF not needed as 1d EMA34 suffices for bias.

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
    
    # Load 1d data ONCE before loop for EMA34 trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (H, L, C)
    # Camarilla R3 = C + (H-L) * 1.1/2
    # Camarilla S3 = C - (H-L) * 1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous completed day's levels
    camarilla_high = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_low = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # Volume confirmation: current 4h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_high_aligned[i]) or \
           np.isnan(camarilla_low_aligned[i]) or np.isnan(vol_ma[i]):
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
        breakout_up = curr_high > camarilla_high_aligned[i]  # break above R3
        breakout_down = curr_low < camarilla_low_aligned[i]  # break below S3
        
        # Trend filter: price relative to EMA34(1d)
        above_ema = curr_close > ema_34_aligned[i]
        below_ema = curr_close < ema_34_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3 AND above EMA34 AND volume confirmation
            if (breakout_up and 
                above_ema and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3 AND below EMA34 AND volume confirmation
            elif (breakout_down and 
                  below_ema and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below S3 (stoploss) OR price crosses below EMA34
            if (curr_low < camarilla_low_aligned[i] or 
                curr_close < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (stoploss) OR price crosses above EMA34
            if (curr_high > camarilla_high_aligned[i] or 
                curr_close > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals