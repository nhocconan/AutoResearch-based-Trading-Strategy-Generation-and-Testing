#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: Trade 4h timeframe using Camarilla R3/S3 breakouts for entry, 
daily EMA34 for trend filter, daily volume spike (>2.0x 20-bar MA) for confirmation, 
and daily choppiness regime filter (CHOP > 61.8 = range, < 38.2 = trending). 
Enter long when price > Camarilla R3 AND above daily EMA34 AND volume spike AND trending regime. 
Enter short when price < Camarilla S3 AND below daily EMA34 AND volume spike AND trending regime. 
Exit on opposite Camarilla level touch (R3/S3) or trend reversal (price crosses EMA34). 
Uses discrete sizing 0.30 to balance return and drawdown. Target 20-50 trades/year on 4h timeframe. 
Works in bull/bear via Camarilla structure (static support/resistance) and trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot points (daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 4.0
    camarilla_s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align Camarilla levels to 4h timeframe (completed daily bar only)
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Get 1d data for daily EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1d data for daily volume spike detection
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Get 1d data for daily choppiness regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n) / (HHV(high,14) - LLV(low,14)))
    # Simplified: CHOP > 61.8 = range, CHOP < 38.2 = trending
    tr1 = np.maximum(high_1d - low_1d, 
                     np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(sum_atr14 / np.log10(14) / (hh14 - ll14 + 1e-10))
    chop_regime_trending = chop_1d < 38.2  # trending regime
    chop_regime_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_trending)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34), volume MA (20), ATR (14), CHOP (14+14)
    start_idx = max(34, 20, 14, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_1d_aligned[i]) or np.isnan(camarilla_s3_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(chop_regime_trending_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above Camarilla R3 AND above daily EMA34 AND volume spike AND trending regime
            long_setup = (close[i] > camarilla_r3_1d_aligned[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_spike_1d_aligned[i] and \
                         chop_regime_trending_aligned[i]
            # Short: price below Camarilla S3 AND below daily EMA34 AND volume spike AND trending regime
            short_setup = (close[i] < camarilla_s3_1d_aligned[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
                          volume_spike_1d_aligned[i] and \
                          chop_regime_trending_aligned[i]
            
            if long_setup:
                signals[i] = 0.30
                position = 1
            elif short_setup:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit: price touches Camarilla S3 OR closes below daily EMA34
            if (close[i] <= camarilla_s3_1d_aligned[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit: price touches Camarilla R3 OR closes above daily EMA34
            if (close[i] >= camarilla_r3_1d_aligned[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0