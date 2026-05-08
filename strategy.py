#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA34 trend filter
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_12h = (close_12h > ema34_12h).astype(float)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Calculate Camarilla pivot levels from previous day
    # Using previous day's high, low, close
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    prev_close = pd.Series(close).shift(1).values
    
    # Camarilla R3 and S3 levels
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Volume spike detection: current volume > 1.5 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above Camarilla R3 with volume spike and 12h uptrend
            long_cond = (close[i] > camarilla_r3[i] and vol_spike[i] and trend_12h_aligned[i] > 0.5)
            
            # Short entry: price breaks below Camarilla S3 with volume spike and 12h downtrend
            short_cond = (close[i] < camarilla_s3[i] and vol_spike[i] and trend_12h_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Camarilla S3 (mean reversion)
            if close[i] < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Camarilla R3 (mean reversion)
            if close[i] > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout with volume confirmation and 12h trend filter on 4h timeframe.
# Works in bull markets (breakouts continue) and bear markets (mean reversion at opposite level).
# Uses 12h EMA34 for trend filter to avoid counter-trend trades.
# Volume spike ensures momentum behind breakouts.
# Target: 20-40 trades/year to minimize fee decay while capturing significant moves.
# Camarilla levels provide precise intraday support/resistance levels that work well in crypto.