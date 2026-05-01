#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout + 1d EMA34 trend + volume spike (>2x 20-bar MA)
# Camarilla R3/S3 levels act as strong intraday support/resistance; breakouts with volume and 1d trend filter capture momentum.
# Works in bull markets via breakouts above R3 in uptrend, and in bear markets via breakdowns below S3 in downtrend.
# Uses discrete sizing (0.30) to minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla levels from prior 1d OHLC (R3, S3)
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    camarilla_R3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low'])
    camarilla_S3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low'])
    
    # Align Camarilla levels to 12h timeframe (use prior day's levels for current bar)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3.values)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3.values)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, 20)  # Need 34 for EMA, 20 for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, above 1d EMA34, and volume confirmation
            if curr_close > camarilla_R3_aligned[i] and curr_close > ema_34_aligned[i] and vol_confirm:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Camarilla S3, below 1d EMA34, and volume confirmation
            elif curr_close < camarilla_S3_aligned[i] and curr_close < ema_34_aligned[i] and vol_confirm:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price breaking below Camarilla S3 or below 1d EMA34
            if curr_close < camarilla_S3_aligned[i] or curr_close < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit on price breaking above Camarilla R3 or above 1d EMA34
            if curr_close > camarilla_R3_aligned[i] or curr_close > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals