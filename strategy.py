#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions; 1d EMA ensures we trade with the higher timeframe trend
# Volume spike (>2.0x 20-period EMA) confirms institutional participation at turning points
# Designed for low trade frequency: ~12-25 trades/year per symbol with 0.25 sizing
# Works in bull/bear: EMA filter avoids counter-trend trades, volume confirms reversal validity

name = "6h_WilliamsR_MeanRev_1dEMA_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Williams %R(14) from 6h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(lookback, 40)  # Need Williams %R and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(williams_r[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of 1d EMA
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:  # Flat - look for mean reversion entries
            if volume_spike[i]:
                # Long: Oversold (%R < -80) in uptrend
                if williams_r[i] < -80 and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Overbought (%R > -20) in downtrend
                elif williams_r[i] > -20 and downtrend:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # No volume confirmation
        
        elif position == 1:  # Long position
            # Exit: price crosses above 1d EMA or %R reaches overbought
            if close[i] >= ema_34_aligned[i] or williams_r[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses below 1d EMA or %R reaches oversold
            if close[i] <= ema_34_aligned[i] or williams_r[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals