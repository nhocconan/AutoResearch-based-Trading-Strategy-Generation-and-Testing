#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d volume spike and 1w trend filter
# - Long when Williams %R(14) < -80 (oversold) AND 1d volume > 1.5x 20-period average AND 1w close > 1w EMA20 (uptrend)
# - Short when Williams %R(14) > -20 (overbought) AND 1d volume > 1.5x 20-period average AND 1w close < 1w EMA20 (downtrend)
# - Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Williams %R identifies exhaustion points in ranging markets
# - Volume spike confirms participation
# - Weekly EMA filter ensures alignment with higher timeframe trend
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_1w_williamsr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h Williams %R(14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.full_like(close, np.nan, dtype=float)
    for i in range(13, n):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Pre-compute 1w EMA20
    close_1w = df_1w['close'].values
    ema_20_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[:20])  # SMA seed
        multiplier = 2 / (20 + 1)
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] * multiplier) + (ema_20_1w[i-1] * (1 - multiplier))
    
    # Align HTF indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, prices, williams_r)  # same timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.5x average)
        vol_ma_12h = np.full_like(prices['volume'].values, np.nan, dtype=float)
        vol_series = prices['volume'].values
        for j in range(19, i+1):
            vol_ma_12h[j] = np.mean(vol_series[j-19:j+1])
        vol_spike = not np.isnan(vol_ma_12h[i]) and vol_series[i] > 1.5 * vol_ma_12h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: oversold AND volume spike AND 1w uptrend
            if (williams_r_aligned[i] < -80 and vol_spike and close[i] > ema_20_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: overbought AND volume spike AND 1w downtrend
            elif (williams_r_aligned[i] > -20 and vol_spike and close[i] < ema_20_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Williams %R crosses -50 (mean reversion completion)
            exit_long = (position == 1 and williams_r_aligned[i] > -50)
            exit_short = (position == -1 and williams_r_aligned[i] < -50)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals