#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla pivots identify key intraday support/resistance levels. Breakout above R3 or below S3
# with 4h trend alignment and volume spike captures strong momentum moves. Designed for 1h timeframe
# with tight entry conditions to achieve 15-37 trades/year, minimizing fee drag while working in
# both bull and bear markets by trading with the higher timeframe trend.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50 = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_4h['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_4h['volume'].values > (2.0 * vol_ema_20)
    
    # Align 4h indicators to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike)
    
    # Calculate daily Camarilla levels (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: H5 = 1.1/12 * (High - Low) + Close, L5 = Close - 1.1/12 * (High - Low)
    # R3 = Close + 1.1/12 * (High - Low) * 1.1, S3 = Close - 1.1/12 * (High - Low) * 1.1
    # Actually, standard Camarilla: R4 = Close + 1.1/12 * (High - Low) * 1.1, R3 = Close + 1.1/12 * (High - Low) * 0.55
    # But we'll use the inner levels: R3 and S3
    camarilla_range = prev_high - prev_low
    r3 = prev_close + (1.1 / 12) * camarilla_range * 0.55
    s3 = prev_close - (1.1 / 12) * camarilla_range * 0.55
    
    # Align daily Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R3 in uptrend with volume spike
            if high[i] > r3_aligned[i] and is_uptrend and volume_spike_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S3 in downtrend with volume spike
            elif low[i] < s3_aligned[i] and is_downtrend and volume_spike_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S3 (reversal to downside)
            if low[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price breaks above R3 (reversal to upside)
            if high[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals