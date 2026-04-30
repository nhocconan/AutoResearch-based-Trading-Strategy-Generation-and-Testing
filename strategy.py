#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation
# Uses discrete sizing 0.20 to balance return and drawdown. Target: 60-150 total trades over 4 years (15-37/year).
# Long when price breaks above Camarilla R3 AND price > 4h EMA34 AND volume spike (>2x 20-period avg).
# Short when price breaks below Camarilla S3 AND price < 4h EMA34 AND volume spike.
# Works in bull via breakout longs, in bear via breakdown shorts. Session filter (08-20 UTC) reduces noise.

name = "1h_Camarilla_R3S3_4hEMA34_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Camarilla levels from previous bar (no look-ahead)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar uses current close as previous
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    camarilla_range = prev_high - prev_low
    camarilla_r3 = prev_close + camarilla_range * 1.1 / 4
    camarilla_s3 = prev_close - camarilla_range * 1.1 / 4
    
    # Calculate 4h EMA(34) for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # need previous bar for Camarilla
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_camarilla_r3 = camarilla_r3[i]
        curr_camarilla_s3 = camarilla_s3[i]
        curr_ema_34_4h = ema_34_4h_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: price breaks above Camarilla R3 AND above 4h EMA34
                if (curr_close > curr_camarilla_r3 and 
                    curr_close > curr_ema_34_4h):
                    signals[i] = 0.20
                    position = 1
                # Bearish entry: price breaks below Camarilla S3 AND below 4h EMA34
                elif (curr_close < curr_camarilla_s3 and 
                      curr_close < curr_ema_34_4h):
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price breaks below Camarilla S3 (mean reversion)
            if curr_close < curr_camarilla_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit when price breaks above Camarilla R3 (mean reversion)
            if curr_close > curr_camarilla_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals