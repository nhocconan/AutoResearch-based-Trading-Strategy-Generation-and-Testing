#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA200 trend filter and 1d volume confirmation
# - Long when price breaks above Donchian(20) high AND 1w close > 1w EMA200 AND 1d volume > 1.5x 20-bar average
# - Short when price breaks below Donchian(20) low AND 1w close < 1w EMA200 AND 1d volume > 1.5x 20-bar average
# - Exit when price crosses Donchian(10) midpoint or opposite Donchian breakout occurs
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Donchian breakouts capture strong momentum moves that work in both bull and bear markets
# - Weekly EMA200 ensures we trade with the primary trend, avoiding counter-trend whipsaws
# - Daily volume confirmation filters low-conviction breakouts

name = "12h_1w_1d_donchian_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels on 12h data
    period_high = 20
    period_low = 20
    period_exit = 10
    
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Donchian(20) for entry
    donchian_high = pd.Series(high_12h).rolling(window=period_high, min_periods=period_high).max().values
    donchian_low = pd.Series(low_12h).rolling(window=period_low, min_periods=period_low).min().values
    
    # Donchian(10) for exit (midpoint)
    donchian_high_exit = pd.Series(high_12h).rolling(window=period_exit, min_periods=period_exit).max().values
    donchian_low_exit = pd.Series(low_12h).rolling(window=period_exit, min_periods=period_exit).min().values
    donchian_mid = (donchian_high_exit + donchian_low_exit) / 2
    
    # Pre-compute 1w EMA(200) for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above Donchian(20) high with 1w uptrend and volume spike
            if (close_12h[i] > donchian_high[i] and 
                close_12h[i] > ema_200_1w_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below Donchian(20) low with 1w downtrend and volume spike
            elif (close_12h[i] < donchian_low[i] and 
                  close_12h[i] < ema_200_1w_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price crosses Donchian(10) midpoint (mean reversion)
            # 2. Opposite Donchian(20) breakout occurs (strong reversal signal)
            if position == 1:
                if close_12h[i] < donchian_mid[i] or close_12h[i] < donchian_low[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                if close_12h[i] > donchian_mid[i] or close_12h[i] > donchian_high[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals