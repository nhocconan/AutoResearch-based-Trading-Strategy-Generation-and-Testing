#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA34 trend filter + volume confirmation
# Williams %R identifies overbought/oversold conditions; extreme readings (>80 or <20) signal potential reversals
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend entries in bear markets
# Volume spike (>1.5 * 20-period EMA) confirms institutional participation at reversal points
# Designed for low trade frequency: ~15-25 trades/year per symbol with 0.25 sizing
# Works in bull markets by buying oversold dips in uptrend and selling overbought rallies in uptrend
# Works in bear markets by selling overbought rallies in downtrend and buying oversold dips in downtrend
# Uses actual 1d EMA calculations (not resampled) for HTF trend filter

name = "6h_WilliamsR_1dEMA34_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R calculation (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA (moderate filter to balance trades)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for Williams %R and EMA34
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA34
        bullish_bias = close[i] > ema_34_aligned[i]
        bearish_bias = close[i] < ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: Williams %R oversold (< -80) with volume spike in uptrend
                if williams_r[i] < -80.0 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: Williams %R overbought (> -20) with volume spike in downtrend
                if williams_r[i] > -20.0 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around EMA34
        
        elif position == 1:  # Long position
            # Exit: Williams %R overbought (> -20) or price below 1d EMA34
            if williams_r[i] > -20.0 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R oversold (< -80) or price above 1d EMA34
            if williams_r[i] < -80.0 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals