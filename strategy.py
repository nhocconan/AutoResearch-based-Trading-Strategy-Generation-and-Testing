#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d EMA34 trend filter and volume spike confirmation.
# Williams %R identifies overbought/oversold conditions. In strong trends (1d EMA34), 
# extreme readings often precede continuations rather than reversals. We fade extremes 
# only when counter to the 1d trend, expecting mean reversion within the 6h timeframe.
# Volume spike confirms participation. Designed for low frequency (20-60 trades/year) 
# to avoid fee drag. Works in bull/bear via 1d EMA34 trend filter.

name = "6h_WilliamsR_Extreme_1dEMA34_TrendFade_VolumeSpike_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA34 trend filter and Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback as standard
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - df_1d['close'].values) / (highest_high - lowest_low)) * -100
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # warmup for EMA34 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(williams_r_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_williams_r = williams_r_aligned[i]
        
        # Volume confirmation: volume > 2.0x 20-period average (moderate threshold)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (2.0 * vol_ma_20)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Fade Williams %R extremes: long when oversold (< -80) but only if 1d trend is down (fade the bounce)
            # Short when overbought (> -20) but only if 1d trend is up (fade the pullback)
            if (curr_williams_r < -80 and 
                curr_close < curr_ema_34_1d and  # 1d downtrend
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif (curr_williams_r > -20 and 
                  curr_close > curr_ema_34_1d and  # 1d uptrend
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when Williams %R returns to neutral territory or price reaches EMA
            if curr_williams_r > -50 or curr_close >= curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R returns to neutral territory or price reaches EMA
            if curr_williams_r < -50 or curr_close <= curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals