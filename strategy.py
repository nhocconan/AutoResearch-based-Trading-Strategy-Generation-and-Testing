#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 12h trend filter and volume confirmation
# - Williams %R(14) from 4h: long when < -80 (oversold), short when > -20 (overbought)
# - 12h EMA(50) trend filter: only long when price > EMA50, short when price < EMA50
# - Volume confirmation: current 4h volume > 2.0x 20-period average to avoid false signals
# - Fixed profit target: exit at 50% of Donchian(20) range to lock in gains
# - Designed for 4h timeframe: targets 30-60 trades/year to balance opportunity and cost
# - Works in bull/bear markets: 12h EMA filter ensures we trade with higher timeframe trend
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "4h_12h_williamsr_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Pre-compute 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Pre-compute 4h Williams %R(14)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_14 - close_4h) / (highest_14 - lowest_14 + 1e-10)
    
    # Pre-compute 4h Donchian(20) for profit target
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2.0
    
    # Pre-compute 4h volume confirmation
    volume_4h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches midpoint of Donchian channel (50% profit target)
            if close_4h[i] >= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches midpoint of Donchian channel (50% profit target)
            if close_4h[i] <= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extreme with trend and volume filters
            if vol_spike[i]:
                # Long: oversold (%R < -80) and price above 12h EMA50 (uptrend)
                if williams_r[i] < -80 and close_4h[i] > ema_50_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: overbought (%R > -20) and price below 12h EMA50 (downtrend)
                elif williams_r[i] > -20 and close_4h[i] < ema_50_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals