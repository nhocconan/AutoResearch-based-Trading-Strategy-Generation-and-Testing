#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Camarilla pivot breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above weekly Camarilla R3 level AND price > 1w EMA50 AND volume > 2.0 * avg_volume(20) on 1d
# Short when price breaks below weekly Camarilla S3 level AND price < 1w EMA50 AND volume > 2.0 * avg_volume(20) on 1d
# Exit when price returns to weekly Camarilla R2/S2 levels (mean reversion) OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Weekly Camarilla provides structural support/resistance levels from higher timeframe
# 1w EMA50 filters for primary trend alignment to avoid counter-trend trades
# Volume spike confirms breakout strength and reduces false signals
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:  # Need at least one completed weekly bar
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (based on previous week's OHLC)
    # Camarilla R3 = close + (high - low) * 1.1/2
    # Camarilla S3 = close - (high - low) * 1.1/2
    # Camarilla R2 = close + (high - low) * 1.1/4
    # Camarilla S2 = close - (high - low) * 1.1/4
    hl_range_1w = high_1w - low_1w
    camarilla_r3 = close_1w + hl_range_1w * 1.1 / 2
    camarilla_s3 = close_1w - hl_range_1w * 1.1 / 2
    camarilla_r2 = close_1w + hl_range_1w * 1.1 / 4
    camarilla_s2 = close_1w - hl_range_1w * 1.1 / 4
    
    # Align weekly Camarilla levels to daily timeframe (wait for completed weekly bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s2)
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Camarilla R3, above 1w EMA50, volume confirmation, in session
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Camarilla S3, below 1w EMA50, volume confirmation, in session
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to weekly Camarilla R2 (mean reversion) OR volume drops below average
            if close[i] < camarilla_r2_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to weekly Camarilla S2 (mean reversion) OR volume drops below average
            if close[i] > camarilla_s2_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals