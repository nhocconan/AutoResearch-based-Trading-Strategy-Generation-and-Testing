#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume spike confirmation
# - Uses 6h Williams %R(14) for mean reversion entries: long when %R < -80, short when %R > -20
# - 1d EMA(50) trend filter: only long when close > EMA50, short when close < EMA50
# - Volume confirmation: require volume > 2.0 * 20-period volume average to avoid false signals
# - Williams %R is effective in ranging markets which dominate 2025 BTC/ETH action
# - Trend filter prevents counter-trend trades during strong moves
# - Volume spike ensures participation and reduces whipsaws
# - Target: 12-25 trades/year on 6h timeframe (50-100 total over 4 years) to minimize fee drag
# - Discrete position sizing: 0.25 for clarity and low turnover

name = "6h_1d_williamsr_meanrev_trend_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 6h Williams %R(14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0, -100 * (highest_high - close) / denominator, -50)
    
    # Pre-compute volume confirmation: volume > 2.0 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: mean reversion or trend change
            if williams_r[i] > -20:  # Williams %R exits overbought
                position = 0
                signals[i] = 0.0
            elif close[i] < ema_50_aligned[i]:  # Trend filter exit
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: mean reversion or trend change
            if williams_r[i] < -80:  # Williams %R exits oversold
                position = 0
                signals[i] = 0.0
            elif close[i] > ema_50_aligned[i]:  # Trend filter exit
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries with trend and volume confirmation
            if williams_r[i] < -80 and close[i] > ema_50_aligned[i] and volume_confirm[i]:
                position = 1
                signals[i] = 0.25
            elif williams_r[i] > -20 and close[i] < ema_50_aligned[i] and volume_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals