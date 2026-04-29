#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Long when: price breaks above R3 (Camarilla resistance) AND price > 1w EMA50 (uptrend) AND volume > 1.5x 20-period average volume
# Short when: price breaks below S3 (Camarilla support) AND price < 1w EMA50 (downtrend) AND volume > 1.5x 20-period average volume
# Uses Camarilla for institutional pivot levels, 1w EMA for primary trend, volume spike for conviction.
# Discrete sizing (0.25) minimizes fee churn. Works in bull/bear via trend filter + mean reversion at extremes.
# Timeframe: 1d (primary), HTF: 1w for EMA50 trend.

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1w EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Camarilla pivot levels on 1d (using prior day's OHLC)
    # Camarilla: P = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_pivot = (high + low + close) / 3.0
    camarilla_r3 = close + (high - low) * 1.1 / 2.0
    camarilla_s3 = close - (high - low) * 1.1 / 2.0
    
    # Shift by 1 to use prior day's levels (no look-ahead)
    camarilla_pivot = np.roll(camarilla_pivot, 1)
    camarilla_r3 = np.roll(camarilla_r3, 1)
    camarilla_s3 = np.roll(camarilla_s3, 1)
    camarilla_pivot[0] = np.nan
    camarilla_r3[0] = np.nan
    camarilla_s3[0] = np.nan
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for 1w EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if 1w EMA50 not available
        if np.isnan(ema50_1w_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema50 = ema50_1w_aligned[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        curr_volume_spike = volume_spike[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price falls below camarilla pivot (mean reversion)
            # 2. Price falls below 1w EMA50 (trend change)
            if (curr_close < camarilla_pivot[i] or curr_close < curr_ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price rises above camarilla pivot (mean reversion)
            # 2. Price rises above 1w EMA50 (trend change)
            if (curr_close > camarilla_pivot[i] or curr_close > curr_ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND above 1w EMA50 AND volume spike
            if (curr_close > curr_r3 and curr_close > curr_ema50 and curr_volume_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 AND below 1w EMA50 AND volume spike
            elif (curr_close < curr_s3 and curr_close < curr_ema50 and curr_volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals