#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# - Camarilla R3/S3 levels act as intraday support/resistance on 4h chart
# - Breakout above R3 (bullish) or below S3 (bearish) with volume confirms momentum
# - 1d EMA34 filter ensures we only trade in direction of higher timeframe trend
# - Session filter (08-20 UTC) reduces noise during low-liquidity hours
# - Discrete position sizing (0.20) minimizes fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) for 1h timeframe
# - Works in bull markets via buying breakouts in uptrend and bear markets via selling breakdowns in downtrend

name = "1h_Camarilla_R3S3_Breakout_4hDirection_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 4h data (using previous 4h bar's OHLC)
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Using previous completed 4h bar to avoid look-ahead
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    cam_r3_4h = c_4h + 1.1 * (h_4h - l_4h) * 1.1 / 4
    cam_s3_4h = c_4h - 1.1 * (h_4h - l_4h) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar to close)
    cam_r3_4h_aligned = align_htf_to_ltf(prices, df_4h, cam_r3_4h)
    cam_s3_4h_aligned = align_htf_to_ltf(prices, df_4h, cam_s3_4h)
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(cam_r3_4h_aligned[i]) or np.isnan(cam_s3_4h_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price > Camarilla R3 + 1d uptrend + volume spike
            if (close[i] > cam_r3_4h_aligned[i] and 
                close[i] > ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price < Camarilla S3 + 1d downtrend + volume spike
            elif (close[i] < cam_s3_4h_aligned[i] and 
                  close[i] < ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Camarilla S3 or 1d trend breaks
            if close[i] < cam_s3_4h_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price > Camarilla R3 or 1d trend breaks
            if close[i] > cam_r3_4h_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals