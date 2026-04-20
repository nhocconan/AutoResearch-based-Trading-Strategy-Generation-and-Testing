# Hypothesis: 12h Williams %R mean reversion with volume confirmation and 1d trend filter.
# Williams %R identifies overbought/oversold conditions. In trending markets (1d EMA50),
# we take mean-reversion entries against the short-term extreme but with the 1d trend.
# Volume confirms conviction. Designed for fewer trades (target: 20-50/year) to avoid fee drag.
# Works in bull/bear: mean reversion in trends performs well in both regimes.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Daily EMA(50) for trend filter
    ema50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Williams %R parameters
    williams_period = 14
    overbought = -20
    oversold = -80
    
    for i in range(100, n):
        # Skip if NaN in trend filter
        if np.isnan(ema50_daily_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Williams %R for period ending at i
        start_idx = max(0, i - williams_period + 1)
        highest_high = np.max(high[start_idx:i+1])
        lowest_low = np.min(low[start_idx:i+1])
        
        if highest_high == lowest_low:
            williams_r = -50  # neutral if no range
        else:
            williams_r = (highest_high - close[i]) / (highest_high - lowest_low) * -100
        
        ema50_daily = ema50_daily_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_ma_recent = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_ok = vol_current > 1.5 * vol_ma_recent
        
        # Trend filter
        uptrend = close[i] > ema50_daily
        downtrend = close[i] < ema50_daily
        
        if position == 0:
            # Long: oversold + uptrend + volume confirmation
            if williams_r < oversold and uptrend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: overbought + downtrend + volume confirmation
            elif williams_r > overbought and downtrend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns to neutral (-50) or trend breaks
            if williams_r >= -50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns to neutral (-50) or trend breaks
            if williams_r <= -50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_WilliamsR_MeanReversion_Trend_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0