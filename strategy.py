#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
# Uses Camarilla R3/S3 levels from 1d data
# Long when price breaks above R3 AND 1d EMA50 uptrend + volume spike
# Short when price breaks below S3 AND 1d EMA50 downtrend + volume spike
# Exit when price returns to pivot point (PP) or opposite Camarilla level
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "12h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels from 1d data (using previous day's OHLC)
    # Camarilla levels: based on previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].shift(1).values  # previous day close
    
    # Calculate pivot point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d_prev) / 3
    
    # Calculate range = H - L
    range_1d = high_1d - low_1d
    
    # Camarilla levels:
    # R4 = PP + range * 1.1/2
    # R3 = PP + range * 1.1/4
    # R2 = PP + range * 1.1/6
    # R1 = PP + range * 1.1/12
    # S1 = PP - range * 1.1/12
    # S2 = PP - range * 1.1/6
    # S3 = PP - range * 1.1/4
    # S4 = PP - range * 1.1/2
    r3 = pp + range_1d * 1.1 / 4
    s3 = pp - range_1d * 1.1 / 4
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # warmup for volume MA and EMA50
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_pp = pp_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema50 = ema50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: price breaks above R3 AND bullish regime
                if curr_close > curr_r3 and curr_close > curr_ema50:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below S3 AND bearish regime
                elif curr_close < curr_s3 and curr_close < curr_ema50:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to pivot point OR breaks below S3
            if curr_close <= curr_pp or curr_close < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to pivot point OR breaks above R3
            if curr_close >= curr_pp or curr_close > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals