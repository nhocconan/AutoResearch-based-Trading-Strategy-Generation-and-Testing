#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels provide high-probability reversal/breakout zones. R3/S3 represent
# strong support/resistance. Breakouts above R3 in uptrend (price > EMA34) or below S3 in
# downtrend capture strong moves with controlled frequency. Volume spike confirms conviction.
# Designed for 20-50 trades/year on 4h to minimize fee drag while maintaining edge in bull/bear markets.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (R3, S3) using previous day's OHLC
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 20-period EMA
        vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1] if i >= 19 else volume[i]
        volume_spike = volume[i] > (2.0 * vol_ema_20)
        
        breakout_long = close[i] > camarilla_r3_aligned[i] and volume_spike
        breakout_short = close[i] < camarilla_s3_aligned[i] and volume_spike
        
        if position == 0:
            # Long: break above R3 in 1d uptrend with volume spike
            if breakout_long and ema_34_1d_aligned[i] < close[i]:  # Uptrend: price > EMA
                signals[i] = 0.25
                position = 1
            # Short: break below S3 in 1d downtrend with volume spike
            elif breakout_short and ema_34_1d_aligned[i] > close[i]:  # Downtrend: price < EMA
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below R3 or loses 1d uptrend
            if close[i] < camarilla_r3_aligned[i] or ema_34_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above S3 or loses 1d downtrend
            if close[i] > camarilla_s3_aligned[i] or ema_34_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals