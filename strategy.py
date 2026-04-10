#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal + 1d EMA trend filter + volume confirmation
# - Long when Williams %R(14) crosses above -80 (oversold bounce) AND price > 1d EMA(50) AND volume > 1.3x 20-period average
# - Short when Williams %R(14) crosses below -20 (overbought rejection) AND price < 1d EMA(50) AND volume > 1.3x 20-period average
# - Exit when Williams %R crosses opposite extreme (-20 for long, -80 for short) OR opposite signal occurs
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Williams %R captures mean reversion in extreme conditions
# - 1d EMA filter ensures we trade with the higher timeframe trend
# - Volume confirmation reduces false reversals
# - Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend)

name = "4h_1d_williamsr_ema_volume_v1"
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
    
    # Pre-compute 4h Williams %R (14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    williams_r = williams_r.values
    
    # Williams %R cross above -80 (oversold bounce)
    williams_r_above_80 = williams_r > -80
    williams_r_below_80_prev = np.concatenate(([False], williams_r[:-1] <= -80))
    williams_r_cross_above_80 = williams_r_above_80 & williams_r_below_80_prev
    
    # Williams %R cross below -20 (overbought rejection)
    williams_r_below_20 = williams_r < -20
    williams_r_above_20_prev = np.concatenate(([True], williams_r[:-1] >= -20))
    williams_r_cross_below_20 = williams_r_below_20 & williams_r_above_20_prev
    
    # Pre-compute 4h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R crosses above -80 AND price > 1d EMA(50) AND volume spike
            if (williams_r_cross_above_80[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R crosses below -20 AND price < 1d EMA(50) AND volume spike
            elif (williams_r_cross_below_20[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Williams %R crosses opposite extreme OR opposite signal occurs
            exit_long = (position == 1 and 
                        (williams_r[i] < -20 or williams_r_cross_below_20[i]))
            exit_short = (position == -1 and 
                         (williams_r[i] > -80 or williams_r_cross_above_80[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals