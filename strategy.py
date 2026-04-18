# 12h_Camarilla_R1S1_Breakout_1dVol_Trend
# Hypothesis: 12h strategy using Camarilla R1/S1 pivot breakouts with 1d volume confirmation and 1d EMA trend filter.
# Camarilla pivots identify key support/resistance levels where breakouts signal strong momentum.
# Volume confirms conviction, EMA filter ensures trades align with daily trend.
# Works in bull markets (break above R1 in uptrend) and bear markets (break below S1 in downtrend).
# Target: 20-40 trades/year to minimize fee drag while capturing high-probability moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Camarilla pivot levels from previous day
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(high[i-1]) and not np.isnan(low[i-1]) and not np.isnan(close[i-1]):
            camarilla_r1[i] = close[i-1] + 1.1 * (high[i-1] - low[i-1]) / 12
            camarilla_s1[i] = close[i-1] - 1.1 * (high[i-1] - low[i-1]) / 12
    
    # Get 1d data for EMA(34) trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA(34) on 1d close
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2/35) + (ema_34_1d[i-1] * 33/35)
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_ma_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_ma_1d[i] = (volume_1d[i] * 2/21) + (vol_ma_1d[i-1] * 19/21)
    
    # Align 1d indicators to 12h timeframe
    ema_34_1d_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_1d_12h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(1, 34, 20)  # need Camarilla, EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema_34_1d_12h[i]) or np.isnan(vol_ma_1d_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5 * 20-period 1d average
        # Note: Using 12h volume vs 1d MA requires scaling - using 2x threshold as approximation
        vol_confirmed = volume[i] > 2.0 * vol_ma_1d_12h[i]
        
        # Trend filter: price above/below 1d EMA34
        trend_up = close[i] > ema_34_1d_12h[i]
        trend_down = close[i] < ema_34_1d_12h[i]
        
        if position == 0:
            # Long entry: close above R1 with volume and uptrend
            if (close[i] > camarilla_r1[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: close below S1 with volume and downtrend
            elif (close[i] < camarilla_s1[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: close below S1 (reversal signal) or trend change
            if close[i] < camarilla_s1[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above R1 (reversal signal) or trend change
            if close[i] > camarilla_r1[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dVol_Trend"
timeframe = "12h"
leverage = 1.0