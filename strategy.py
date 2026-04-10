#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high AND 1w EMA50 rising AND volume > 1.5x 20-bar average
# - Short when price breaks below Donchian(20) low AND 1w EMA50 falling AND volume > 1.5x 20-bar average
# - Exit when price crosses Donchian(10) midpoint (mean reversion) or opposite breakout occurs
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~15-25 trades/year (60-100 total over 4 years) to avoid fee drag
# - Donchian breakouts capture strong momentum moves in both bull and bear markets
# - 1w trend filter ensures we trade with higher timeframe momentum
# - Volume confirmation filters false breakouts
# - Works across BTC/ETH/SOL as structure-based strategy

name = "1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period) on 1d data
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    high_10 = prices['high'].rolling(window=10, min_periods=10).max().values
    low_10 = prices['low'].rolling(window=10, min_periods=10).min().values
    donchian_mid_20 = (high_20 + low_20) / 2.0
    donchian_mid_10 = (high_10 + low_10) / 2.0
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 1w EMA(50) slope for trend direction
    ema_slope = np.zeros_like(ema_50_1w_aligned)
    ema_slope[1:] = ema_50_1w_aligned[1:] - ema_50_1w_aligned[:-1]
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_slope[i]) or
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above Donchian(20) high with rising 1w EMA and volume spike
            if (prices['close'].iloc[i] > high_20[i] and 
                ema_slope[i] > 0 and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below Donchian(20) low with falling 1w EMA and volume spike
            elif (prices['close'].iloc[i] < low_20[i] and 
                  ema_slope[i] < 0 and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price crosses Donchian(10) midpoint (mean reversion)
            # 2. Opposite Donchian(20) breakout occurs
            if position == 1:  # Long position
                if (prices['close'].iloc[i] < donchian_mid_10[i] or 
                    prices['close'].iloc[i] < low_20[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] > donchian_mid_10[i] or 
                    prices['close'].iloc[i] > high_20[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals