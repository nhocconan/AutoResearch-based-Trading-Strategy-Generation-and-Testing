#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and ADX trend filter
# - Long when price breaks above Camarilla R4 (1d) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average
# - Short when price breaks below Camarilla S4 (1d) AND 1d ADX > 25 AND volume > 1.5x 20-period average
# - Exit when price crosses back inside the Camarilla R3-S3 range (mean reversion zone)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Camarilla pivots identify key intraday support/resistance levels derived from prior day
# - ADX filter ensures we trade only when there is sufficient trend strength
# - Volume confirmation reduces false breakouts
# - R3-S3 exit provides mean reversion logic in ranging markets

name = "6h_1d_camarilla_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 6h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d Camarilla levels (based on prior day OHLC)
    # Camarilla: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4)
    #          S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    camarilla_r4 = np.zeros(len(df_1d))
    camarilla_r3 = np.zeros(len(df_1d))
    camarilla_s3 = np.zeros(len(df_1d))
    camarilla_s4 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        h = df_1d['high'].iloc[i]
        l = df_1d['low'].iloc[i]
        c = df_1d['close'].iloc[i]
        camarilla_r4[i] = c + ((h - l) * 1.1 / 2)
        camarilla_r3[i] = c + ((h - l) * 1.1 / 4)
        camarilla_s3[i] = c - ((h - l) * 1.1 / 4)
        camarilla_s4[i] = c - ((h - l) * 1.1 / 2)
    
    # Pre-compute 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    tr_14 = np.zeros_like(tr)
    dm_plus_14 = np.zeros_like(dm_plus)
    dm_minus_14 = np.zeros_like(dm_minus)
    
    tr_14[13] = np.mean(tr[1:14])
    dm_plus_14[13] = np.mean(dm_plus[1:14])
    dm_minus_14[13] = np.mean(dm_minus[1:14])
    
    for i in range(14, len(tr)):
        tr_14[i] = (tr_14[i-1] * 13 + tr[i]) / 14
        dm_plus_14[i] = (dm_plus_14[i-1] * 13 + dm_plus[i]) / 14
        dm_minus_14[i] = (dm_minus_14[i-1] * 13 + dm_minus[i]) / 14
    
    # DI+ and DI-
    di_plus = np.where(tr_14 > 0, (dm_plus_14 / tr_14) * 100, 0)
    di_minus = np.where(tr_14 > 0, (dm_minus_14 / tr_14) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = np.zeros_like(dx)
    adx[27] = np.mean(dx[14:28])  # First ADX after 14+14 periods
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # ADX trend filter: trending when ADX > 25
    adx_trending = adx > 25
    
    # Align HTF indicators to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(adx_trending_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla R4 AND trending AND volume spike
            if (close[i] > camarilla_r4_aligned[i] and 
                adx_trending_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla S4 AND trending AND volume spike
            elif (close[i] < camarilla_s4_aligned[i] and 
                  adx_trending_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses back inside Camarilla R3-S3 range (mean reversion zone)
            exit_long = (position == 1 and close[i] < camarilla_r3_aligned[i])
            exit_short = (position == -1 and close[i] > camarilla_s3_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals