#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA200 trend filter + volume confirmation
# - Long when price breaks above 20-period high AND close > 12h EMA200 AND volume > 1.5x 20-period average
# - Short when price breaks below 20-period low AND close < 12h EMA200 AND volume > 1.5x 20-period average
# - Exit when price crosses 10-period EMA in opposite direction
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Donchian breakouts capture momentum in trending markets
# - 12h EMA200 ensures we only trade with the higher timeframe trend
# - Volume confirmation reduces false breakouts

name = "4h_12h_donchian_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian high/low (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 4h 10-period EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Pre-compute 12h EMA200 trend filter
    close_12h = df_12h['close'].values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_10[i]) or 
            np.isnan(ema_200_12h_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Donchian breakout above + 12h EMA200 uptrend + volume spike
            if (close[i] > donch_high[i] and 
                close[i] > ema_200_12h_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Donchian breakdown below + 12h EMA200 downtrend + volume spike
            elif (close[i] < donch_low[i] and 
                  close[i] < ema_200_12h_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price crosses 10-period EMA in opposite direction
            exit_long = (position == 1 and close[i] < ema_10[i])
            exit_short = (position == -1 and close[i] > ema_10[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals