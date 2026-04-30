#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 80-120 total trades over 4 years (20-30/year).
# Camarilla pivot levels provide high-probability reversal/breakout points.
# Breakout above R3 or below S3 with volume confirmation and 1d EMA34 trend alignment.
# Works in bull via breakout longs above R3, in bear via breakdown shorts below S3.
# Volume spike filters for institutional participation, avoids false breakouts.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d EMA(34) for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (H + L + C) / 3
    typical_price = (high + low + close) / 3.0
    # Range = H - L
    rng = high - low
    
    # Camarilla levels (based on previous day's data)
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    prev_typical = np.concatenate([[np.nan], typical_price[:-1]])
    prev_rng = np.concatenate([[np.nan], rng[:-1]])
    
    # Calculate Camarilla R3 and S3 levels
    camarilla_r3 = prev_typical + (prev_rng * 1.1 / 4.0)
    camarilla_s3 = prev_typical - (prev_rng * 1.1 / 4.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(34, 20)  # warmup for EMA34 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: price breaks above R3 AND above 1d EMA34 (bullish bias)
                if (curr_close > curr_r3 and 
                    curr_close > curr_ema_34_1d):
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below S3 AND below 1d EMA34 (bearish bias)
                elif (curr_close < curr_s3 and 
                      curr_close < curr_ema_34_1d):
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: price drops below R3 (failed breakout) OR loses 1d EMA34 trend
            if (curr_close < curr_r3 or  # Failed to hold above R3
                curr_close < curr_ema_34_1d):  # Lost bullish trend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above S3 (failed breakdown) OR loses 1d EMA34 trend
            if (curr_close > curr_s3 or  # Failed to hold below S3
                curr_close > curr_ema_34_1d):  # Lost bearish trend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals