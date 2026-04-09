#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA(50) trend filter and volume confirmation
# - Uses 1w EMA(50) for trend direction (long when price > EMA, short when price < EMA)
# - Uses 12h Donchian(20) channels for breakout entries (long on upper band break, short on lower)
# - Requires volume > 1.5 * 20-period volume average for confirmation
# - Fixed position size 0.25 to manage drawdown and reduce fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Works in bull markets via breakouts above resistance, in bear via breakdowns below support
# - EMA filter prevents counter-trend trades in strong trends

name = "12h_1w_donchian_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Pre-compute 12h Donchian(20) channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper band: highest high of last 20 periods
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian lower band: lowest low of last 20 periods
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute volume confirmation: volume > 1.5 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1w EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions: stoploss or mean reversion
            if close[i] < donchian_lower[i]:  # Exit when price breaks below lower Donchian
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: stoploss or mean reversion
            if close[i] > donchian_upper[i]:  # Exit when price breaks above upper Donchian
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout entries in direction of 1w trend with volume confirmation
            if uptrend and close[i] > donchian_upper[i] and volume_confirm[i]:  # Break above upper band in uptrend
                position = 1
                signals[i] = 0.25
            elif downtrend and close[i] < donchian_lower[i] and volume_confirm[i]:  # Break below lower band in downtrend
                position = -1
                signals[i] = -0.25
    
    return signals