#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Camarilla pivots from 1d provide key support/resistance levels. R3/S3 are strong reversal zones;
# breakouts beyond these levels with volume confirmation indicate momentum continuation.
# 1d EMA34 > 1d EMA89 filter ensures trades only in the direction of the daily trend.
# Volume spike (2.0x 20-period average) confirms breakout validity.
# Discrete sizing 0.25 balances risk and minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d Camarilla pivot levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r3 = pivot + (prev_high - prev_low) * 1.1 / 4.0
    s3 = pivot - (prev_high - prev_low) * 1.1 / 4.0
    
    # Align 1d Camarilla levels to 12h timeframe (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1d EMA34 and EMA89 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89 = pd.Series(df_1d['close']).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1d EMAs to 12h timeframe (wait for completed 1d bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    ema_89_aligned = align_htf_to_ltf(prices, df_1d, ema_89)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 34, 89)  # warmup for volume MA and EMAs
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(ema_89_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema_34 = ema_34_aligned[i]
        curr_ema_89 = ema_89_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish breakout: price breaks above R3 AND daily EMA34 > EMA89 (uptrend)
                if curr_close > curr_r3 and curr_ema_34 > curr_ema_89:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price breaks below S3 AND daily EMA34 < EMA89 (downtrend)
                elif curr_close < curr_s3 and curr_ema_34 < curr_ema_89:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below S3 (mean reversion to support)
            if curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above R3 (mean reversion to resistance)
            if curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals