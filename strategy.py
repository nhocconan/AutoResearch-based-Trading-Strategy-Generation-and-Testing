#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike filter and 1w trend alignment
# - Long when price breaks above Donchian(20) high AND 1d volume > 2.0x 20-bar average AND 1w close > 1w EMA50
# - Short when price breaks below Donchian(20) low AND 1d volume > 2.0x 20-bar average AND 1w close < 1w EMA50
# - Exit when price crosses Donchian(10) midpoint (mean reversion) or opposite breakout occurs
# - Uses discrete position sizing (0.30) to balance return and drawdown
# - Targets ~25-40 trades/year (100-160 total over 4 years) to avoid fee drag
# - Donchian breakouts capture strong momentum moves
# - Volume confirmation ensures breakout validity
# - 1w trend filter ensures we trade with higher timeframe momentum

name = "4h_1d_1w_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels on 4h data
    period_high = 20
    period_low = 20
    period_exit = 10
    
    highest_high = pd.Series(prices['high'].values).rolling(window=period_high, min_periods=period_high).max().values
    lowest_low = pd.Series(prices['low'].values).rolling(window=period_low, min_periods=period_low).min().values
    highest_high_exit = pd.Series(prices['high'].values).rolling(window=period_exit, min_periods=period_exit).max().values
    lowest_low_exit = pd.Series(prices['low'].values).rolling(window=period_exit, min_periods=period_exit).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Pre-compute 1d volume confirmation: > 2.0x 20-period average
    volume_20_avg = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above Donchian(20) high with volume spike and 1w uptrend
            if (prices['close'].iloc[i] > highest_high[i] and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.30
            # Short signal: price breaks below Donchian(20) low with volume spike and 1w downtrend
            elif (prices['close'].iloc[i] < lowest_low[i] and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.30
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price crosses Donchian(10) midpoint (mean reversion)
            # 2. Opposite breakout occurs
            if position == 1:
                if prices['close'].iloc[i] < donchian_mid[i] or prices['close'].iloc[i] < lowest_low_exit[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30  # Hold long
            elif position == -1:
                if prices['close'].iloc[i] > donchian_mid[i] or prices['close'].iloc[i] > highest_high_exit[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30  # Hold short
    
    return signals