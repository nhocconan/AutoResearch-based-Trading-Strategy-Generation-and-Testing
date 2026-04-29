#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout + 12h EMA50 Trend Filter + Volume Spike
# Long when: price breaks above Camarilla R3 (strong resistance) AND price > 12h EMA50 (uptrend) AND volume > 1.5x 20-period avg volume
# Short when: price breaks below Camarilla S3 (strong support) AND price < 12h EMA50 (downtrend) AND volume > 1.5x 20-period avg volume
# Uses Camarilla levels for institutional support/resistance, 12h EMA for trend filter, volume spike for confirmation, discrete sizing (0.25) to minimize fee churn.
# Works in bull/bear via trend filter (avoid counter-trend) + volatility expansion (volume spike) for breakout validity.
# Timeframe: 4h (primary), HTF: 12h for EMA50 trend.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 12h EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from prior day (using daily data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation
    prior_high = pd.Series(df_1d['high'].values).shift(1).values  # shift to use prior day
    prior_low = pd.Series(df_1d['low'].values).shift(1).values
    prior_close = pd.Series(df_1d['close'].values).shift(1).values
    
    # Camarilla levels: R3 = Close + (High - Low) * 1.1/4, S3 = Close - (High - Low) * 1.1/4
    camarilla_r3 = prior_close + (prior_high - prior_low) * 1.1 / 4.0
    camarilla_s3 = prior_close - (prior_high - prior_low) * 1.1 / 4.0
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if Camarilla data not available (first bar has no prior day)
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_camarilla_r3 = camarilla_r3_aligned[i]
        curr_camarilla_s3 = camarilla_s3_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price falls below Camarilla S3 (mean reversion)
            # 2. Price falls below 12h EMA50 (trend change)
            if (curr_close < curr_camarilla_s3 or
                curr_close < curr_ema_50_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price rises above Camarilla R3 (mean reversion)
            # 2. Price rises above 12h EMA50 (trend change)
            if (curr_close > curr_camarilla_r3 or
                curr_close > curr_ema_50_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 AND above 12h EMA50 AND volume spike
            if (curr_close > curr_camarilla_r3 and
                curr_close > curr_ema_50_12h and
                curr_volume_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla S3 AND below 12h EMA50 AND volume spike
            elif (curr_close < curr_camarilla_s3 and
                  curr_close < curr_ema_50_12h and
                  curr_volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals