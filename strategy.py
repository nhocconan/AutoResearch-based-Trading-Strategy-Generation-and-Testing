#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme with 1d EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; extreme readings (>80 or <20) 
# combined with 1d EMA34 trend direction captures mean reversion in trending markets.
# Volume spike (2.0x 20-period average) confirms institutional participation at extremes.
# Works in both bull and bear markets by following the 1d EMA34 trend for direction.
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsR_Extreme_1dEMA34_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA calculation
    
    for i in range(start_idx, n):
        # Get the prior completed 1d bar's OHLC for Williams %R calculation
        o_1d = df_1d['open'].values
        h_1d = df_1d['high'].values
        l_1d = df_1d['low'].values
        c_1d = df_1d['close'].values
        
        # Shift by 1 to use prior completed 1d bar (no look-ahead)
        o_1d_shifted = np.roll(o_1d, 1)
        h_1d_shifted = np.roll(h_1d, 1)
        l_1d_shifted = np.roll(l_1d, 1)
        c_1d_shifted = np.roll(c_1d, 1)
        o_1d_shifted[0] = np.nan
        h_1d_shifted[0] = np.nan
        l_1d_shifted[0] = np.nan
        c_1d_shifted[0] = np.nan
        
        # Align the shifted 1d OHLC to 6h timeframe
        h_1d_aligned = align_htf_to_ltf(prices, df_1d, h_1d_shifted)
        l_1d_aligned = align_htf_to_ltf(prices, df_1d, l_1d_shifted)
        c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d_shifted)
        
        # Calculate Williams %R for prior 1d bar: %R = (H - C)/(H - L) * -100
        prior_high = h_1d_aligned[i]
        prior_low = l_1d_aligned[i]
        prior_close = c_1d_aligned[i]
        
        if np.isnan(prior_high) or np.isnan(prior_low) or np.isnan(prior_close) or prior_high == prior_low:
            signals[i] = 0.0
            continue
            
        williams_r = (prior_high - prior_close) / (prior_high - prior_low) * -100
        
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: Williams %R < -80 (oversold) AND price above 1d EMA34 (uptrend)
                if williams_r < -80 and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R > -20 (overbought) AND price below 1d EMA34 (downtrend)
                elif williams_r > -20 and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Williams %R rises above -50 (momentum weakening) or price falls below 1d EMA34
            if williams_r > -50 or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R falls below -50 (momentum weakening) or price rises above 1d EMA34
            if williams_r < -50 or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals