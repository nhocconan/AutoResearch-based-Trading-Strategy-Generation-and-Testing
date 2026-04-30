#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme readings with 1d EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. Extreme readings below -90 (oversold) or above -10 (overbought)
# combined with 1d EMA34 trend alignment and volume spike captures mean reversion in trending markets.
# Works in bull markets (buying oversold dips in uptrend) and bear markets (selling overbought rallies in downtrend).
# Uses discrete sizing 0.25 to minimize fee churn. Target: 75-200 trades over 4 years.

name = "6h_WilliamsR_Extreme_1dEMA34_Volume_v1"
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
    
    # Calculate 1d Williams %R (14-period) from prior 1d bar
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for Williams %R calculation
        return np.zeros(n)
    
    # Prior 1d OHLC for Williams %R calculation (shift to avoid look-ahead)
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Calculate Williams %R(14) on 1d timeframe
    highest_high = pd.Series(prior_high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(prior_low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - prior_close) / (highest_high - lowest_low)
    
    # Align Williams %R to 6h timeframe (wait for 1d bar to close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 34, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r_aligned[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Williams %R extreme and 1d trend filter
            if curr_volume_spike:
                # Bullish: Williams %R below -90 (oversold) + price above 1d EMA34
                if curr_williams_r < -90 and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Williams %R above -10 (overbought) + price below 1d EMA34
                elif curr_williams_r > -10 and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: Williams %R rises above -50 (exiting oversold) or loses 1d trend
            if curr_williams_r > -50 or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R falls below -50 (exiting overbought) or loses 1d trend
            if curr_williams_r < -50 or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals