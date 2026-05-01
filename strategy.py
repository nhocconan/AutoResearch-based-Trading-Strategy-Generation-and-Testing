#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R breakout with 1w EMA34 trend filter and volume confirmation (>1.5x 20-bar MA)
# Uses 1w HTF for strong trend alignment to avoid whipsaws in ranging markets.
# Williams %R identifies overbought/oversold conditions; breakout from extreme levels with volume confirms momentum.
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).
# Works in bull (trend continuation) and bear (mean reversion from extremes) via trend filter.

name = "12h_WilliamsR_Breakout_1wEMA34_Trend_VolumeConfirm_v1"
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
    
    # 1w HTF data for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA(34) on 1w close
    ema_1w_34 = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 12h timeframe
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # Calculate Williams %R(14) on 12h data
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for volume MA and 14 for Williams %R
    
    for i in range(start_idx, n):
        if np.isnan(ema_1w_34_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 from oversold, above 1w EMA, and volume confirmation
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and curr_close > ema_1w_34_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from overbought, below 1w EMA, and volume confirmation
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and curr_close < ema_1w_34_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R crossing below -50 or price below 1w EMA
            if williams_r[i] < -50 and williams_r[i-1] >= -50 or curr_close < ema_1w_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R crossing above -50 or price above 1w EMA
            if williams_r[i] > -50 and williams_r[i-1] <= -50 or curr_close > ema_1w_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals