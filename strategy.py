#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme with 1d ADX trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. Extreme readings (< -80 or > -20)
# combined with strong 1d trend (ADX > 25) and volume spike provide high-probability
# mean reversion entries in ranging markets and trend continuation in strong markets.
# Designed for 12-30 trades/year on 6h to minimize fee drag while working in both bull/bear.

name = "6h_WilliamsR_Extreme_1dADX25_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend filter
    plus_dm = np.diff(df_1d['high'].values)
    minus_dm = np.diff(df_1d['low'].values)
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr1 = np.diff(df_1d['high'].values)
    tr2 = np.diff(df_1d['low'].values)
    tr3 = np.abs(np.diff(df_1d['close'].values))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with original length
    
    atr_1d = pd.Series(tr).ewm(span=25, adjust=False, min_periods=25).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=25, adjust=False, min_periods=25).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=25, adjust=False, min_periods=25).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=25, adjust=False, min_periods=25).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Williams %R (14-period) on 6h data
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after sufficient warmup for Williams %R
        # Skip if any value is NaN or outside session
        if (np.isnan(adx_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Williams %R using data up to current bar
        lookback = min(14, i+1)
        highest_high = np.max(high[i-lookback+1:i+1])
        lowest_low = np.min(low[i-lookback+1:i+1])
        
        # Avoid division by zero
        if highest_high == lowest_low:
            williams_r = -50.0  # neutral value
        else:
            williams_r = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
        
        # Volume confirmation: current volume > 2x 20-period EMA of volume
        vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1] if i >= 19 else volume[i]
        volume_spike = volume[i] > (2.0 * vol_ema_20)
        
        # Extreme Williams %R conditions with volume spike
        williams_oversold = williams_r < -80 and volume_spike
        williams_overbought = williams_r > -20 and volume_spike
        
        if position == 0:
            # Long: Williams %R oversold in 1d uptrend (ADX > 25) with volume spike
            if williams_oversold and adx_1d_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought in 1d downtrend (ADX > 25) with volume spike
            elif williams_overbought and adx_1d_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns above -50 or loses 1d uptrend
            if williams_r > -50 or adx_1d_aligned[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns below -50 or loses 1d downtrend
            if williams_r < -50 or adx_1d_aligned[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals