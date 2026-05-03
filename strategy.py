#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels identify key support/resistance. Breakouts above R3 or below S3 with
# volume confirmation indicate strong momentum. 12h EMA34 filter ensures alignment with higher
# timeframe trend. Designed for 12-30 trades/year on 6h to minimize fee drag while capturing
# significant moves in both bull and bear markets.

name = "6h_Camarilla_R3_S3_Breakout_12hEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 12h Camarilla pivot levels (using previous 12h bar's OHLC)
    # Camarilla levels: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), 
    # S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # where C = (H+L+O)/3 (typical price)
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    high_12h = df_12h['high']
    low_12h = df_12h['low']
    
    camarilla_r3 = typical_price_12h + ((high_12h - low_12h) * 1.1 / 4)
    camarilla_s3 = typical_price_12h - ((high_12h - low_12h) * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have previous bar data
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 20-period EMA
        if i >= 19:
            vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        if position == 0:
            # Long: break above R3 with volume spike and 12h uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike and 
                ema_34_12h_aligned[i] > close[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume spike and 12h downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike and 
                  ema_34_12h_aligned[i] < close[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below R3 or loses 12h uptrend
            if close[i] < camarilla_r3_aligned[i] or ema_34_12h_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above S3 or loses 12h downtrend
            if close[i] > camarilla_s3_aligned[i] or ema_34_12h_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals