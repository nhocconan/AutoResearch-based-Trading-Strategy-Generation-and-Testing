#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Camarilla pivot levels provide precise support/resistance based on prior day's range
# Breakout above R3 or below S3 with volume confirmation indicates strong momentum
# 4h EMA50 > EMA200 ensures alignment with medium-term trend to avoid whipsaw
# Volume spike (2.0x 24-period average) confirms institutional participation
# Discrete sizing 0.20 minimizes fee churn. Target: 60-150 total trades over 4 years (15-37/year).
# Uses 1h timeframe for entry timing with 4h/1d for signal direction as required.

name = "1h_Camarilla_R3S3_4hEMA50_200_Trend_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA50 and EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h EMAs to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels based on prior day's OHLC
    camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.125 / 2)  # R3 = C + (H-L)*1.125/2
    camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.125 / 2)  # S3 = C - (H-L)*1.125/2
    
    # Align to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 24-period average (24*1h = 1 day)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 24)  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_4h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50 = ema50_4h_aligned[i]
        curr_ema200 = ema200_4h_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and bullish/bearish trend alignment
            if curr_volume_spike:
                # Bullish entry: price > EMA50 > EMA200 and break above R3
                if curr_ema50 > curr_ema200 and curr_close > curr_r3:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price < EMA50 < EMA200 and break below S3
                elif curr_ema50 < curr_ema200 and curr_close < curr_s3:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below R3 (breakout fails) OR trend weakens (EMA50 < EMA200)
            if curr_close < curr_r3 or curr_ema50 < curr_ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit when price rises above S3 (breakdown fails) OR trend weakens (EMA50 > EMA200)
            if curr_close > curr_s3 or curr_ema50 > curr_ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals