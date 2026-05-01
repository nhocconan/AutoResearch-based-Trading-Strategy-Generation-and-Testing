#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d EMA50 trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In ranging markets, reversals from extremes work well.
# Long when Williams %R crosses above -80 from below, price above 1d EMA50, and volume > 1.5x 20-bar MA.
# Short when Williams %R crosses below -20 from above, price below 1d EMA50, and volume > 1.5x 20-bar MA.
# Uses 1d EMA50 for higher-timeframe trend alignment to reduce whipsaws.
# Volume confirmation filters low-participation moves. Discrete sizing (0.25) minimizes fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits.
# Works in both bull (trend filter allows longs in uptrend) and bear (allows shorts in downtrend) markets.

name = "12h_WilliamsR_MeanReversion_1dEMA50_Trend_VolumeConfirm_v1"
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
    
    # 1d HTF data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) on 1d close
    ema_1d_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 12h timeframe
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Williams %R(14) calculation: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for volume MA and Williams %R
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_50_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Williams %R crossovers
        williams_r_cross_above_80 = williams_r[i] > -80 and williams_r[i-1] <= -80
        williams_r_cross_below_20 = williams_r[i] < -20 and williams_r[i-1] >= -20
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 from below, price above 1d EMA50, volume confirm
            if williams_r_cross_above_80 and curr_close > ema_1d_50_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above, price below 1d EMA50, volume confirm
            elif williams_r_cross_below_20 and curr_close < ema_1d_50_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R crossing below -50 (mean reversion) or price below 1d EMA50
            if williams_r[i] < -50 or curr_close < ema_1d_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R crossing above -50 (mean reversion) or price above 1d EMA50
            if williams_r[i] > -50 or curr_close > ema_1d_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals