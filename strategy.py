#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Williams %R(14) from 4h: oversold < -80 for long, overbought > -20 for short
# - 1d EMA(50) trend filter: price above EMA for long bias, below for short bias
# - Volume confirmation: current 4h volume > 1.5x 20-period average
# - Designed for 4h timeframe: targets 20-50 trades/year (80-200 total over 4 years) to avoid fee drag
# - Works in bull/bear markets: daily EMA filter ensures we trade with higher timeframe trend
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "4h_1d_williams_r_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 4h Williams %R(14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams %R calculation
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Pre-compute 4h volume confirmation
    volume = prices['volume'].values
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R returns above -50 (mean reversion complete) or breaks below -90 (stop)
            if williams_r[i] > -50 or williams_r[i] < -90:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R returns below -50 (mean reversion complete) or breaks above -10 (stop)
            if williams_r[i] < -50 or williams_r[i] > -10:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R mean reversion with trend and volume filters
            if vol_spike[i]:
                # Mean reversion long: oversold (< -80) with price above daily EMA (uptrend)
                if williams_r[i] < -80 and prices['close'].iloc[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Mean reversion short: overbought (> -20) with price below daily EMA (downtrend)
                elif williams_r[i] > -20 and prices['close'].iloc[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals