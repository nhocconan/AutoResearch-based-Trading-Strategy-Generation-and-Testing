#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d ADX trend filter and volume confirmation
# Camarilla pivot levels provide high-probability reversal/breakout zones; 1d ADX>25 ensures 
# we only trade in trending markets to avoid whipsaws. Volume spike confirms participation.
# Designed for low trade frequency (12-37/year) on 12h timeframe to minimize fee drag.
# Works in both bull and bear markets by aligning with daily trend and using volatility-based stops.

name = "12h_Camarilla_R3S3_1dADX25_VolumeSpike"
timeframe = "12h"
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
    
    # Get 1d data for ADX and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period) for trend strength filter
    # ADX = 100 * smoothed moving average of |+DI - -DI| / (+DI + -DI)
    # Simplified: use Welles Wilder's ATR-based DX calculation
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = pd.Series(df_1d['low']).diff().abs()
    tr3 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'])).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Align 1d indicators to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Calculate 12h Camarilla levels (based on previous day's OHLC)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6
    #          S2 = C - (H-L)*1.1/6, S1 = C - (H-L)*1.1/4, S3 = C - (H-L)*1.1/2
    # We use R3/S3 as breakout levels
    df_1d_shifted = df_1d.shift(1)  # Use previous day's OHLC
    camarilla_r3 = df_1d_shifted['close'] + (df_1d_shifted['high'] - df_1d_shifted['low']) * 1.1 / 4
    camarilla_s3 = df_1d_shifted['close'] - (df_1d_shifted['high'] - df_1d_shifted['low']) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d_shifted, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d_shifted, camarilla_s3.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend strength: ADX > 25 indicates trending market
        is_trending = adx_aligned[i] > 25
        
        if position == 0:
            # Long: Price breaks above Camarilla R3 in trending market with volume spike
            if high[i] > camarilla_r3_aligned[i] and is_trending and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 in trending market with volume spike
            elif low[i] < camarilla_s3_aligned[i] and is_trending and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below Camarilla S3 (reversal to downside)
            if low[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above Camarilla R3 (reversal to upside)
            if high[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals