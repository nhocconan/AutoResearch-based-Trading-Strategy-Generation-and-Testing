#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high AND 12h close > EMA50 AND volume > 2.0x avg
# - Short when price breaks below Donchian(20) low AND 12h close < EMA50 AND volume > 2.0x avg
# - Exit when price crosses Donchian(20) midpoint OR opposite breakout occurs
# - Uses discrete position sizing (0.25) to control drawdown
# - Targets ~20-50 trades/year (80-200 total over 4 years) to avoid fee drag
# - Donchian channels provide clear structure with proven edge on SOLUSDT
# - 12h EMA50 filter ensures alignment with higher timeframe trend
# - High volume threshold (2.0x) reduces false breakouts

name = "4h_12h_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian(20) channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high: rolling max of high over 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of low over 20 periods
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: average of high and low channels
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Pre-compute 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Pre-compute volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above Donchian high AND 12h uptrend AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i-1] <= donchian_high[i-1] and  # Ensure breakout just occurred
                close[i] > ema50_12h_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below Donchian low AND 12h downtrend AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i-1] >= donchian_low[i-1] and  # Ensure breakdown just occurred
                  close[i] < ema50_12h_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price crosses Donchian midpoint (trend change)
            # 2. Opposite breakout occurs (strong reversal signal)
            if position == 1:  # Long position
                if (close[i] < donchian_mid[i] or 
                    (close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1])):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (close[i] > donchian_mid[i] or 
                    (close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1])):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals