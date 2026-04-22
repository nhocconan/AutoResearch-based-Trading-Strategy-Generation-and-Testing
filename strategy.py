#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 Breakout with 1w EMA50 trend filter and volume spike
# Uses Camarilla pivot levels from previous day (R1, S1) for breakout entries
# Long when price breaks above R1 with 1w uptrend and volume spike
# Short when price breaks below S1 with 1w downtrend and volume spike
# Weekly trend filter provides stronger trend bias, reducing whipsaws in choppy markets
# Designed for 4h timeframe to target 25-40 trades/year per symbol.
# Weekly trend filter should work better in both bull and bear markets by filtering counter-trend noise.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Load 1d data for pivot calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    camarilla_R1_1d = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12
    camarilla_S1_1d = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1_1d)
    camarilla_S1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1_1d)
    
    # 1w EMA(50) for higher timeframe trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike filter (20-period on 4h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_R1_1d_aligned[i]) or np.isnan(camarilla_S1_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 1w uptrend + volume spike
            if (close[i] > camarilla_R1_1d_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + 1w downtrend + volume spike
            elif (close[i] < camarilla_S1_1d_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to Camarilla center (previous close) or trend reversal
            camarilla_center_1d = prev_close_1d
            camarilla_center_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_center_1d)
            
            if position == 1:
                # Exit on price below camarilla center or trend reversal
                if (close[i] < camarilla_center_1d_aligned[i] or 
                    close[i] < ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on price above camarilla center or trend reversal
                if (close[i] > camarilla_center_1d_aligned[i] or 
                    close[i] > ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0