#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w trend filter and volume confirmation
# - Long when Williams %R(14) crosses above -80 (oversold) with volume > 1.3x 20-bar avg AND 1w close > 1w EMA34
# - Short when Williams %R(14) crosses below -20 (overbought) with volume > 1.3x 20-bar avg AND 1w close < 1w EMA34
# - Exit when Williams %R returns to -50 midpoint
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~20 trades/year (80 total over 4 years) to avoid fee drag
# - 1w trend filter ensures we only trade with the higher timeframe trend
# - Volume confirmation ensures institutional participation in reversals
# - Williams %R is effective in both bull and bear markets for identifying exhaustion points

name = "1d_williamsr_meanreversion_1w_trend_volume_v1"
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
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    
    # 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Pre-compute 1d Williams %R(14)
    highest_high_14 = prices['high'].rolling(window=14, min_periods=14).max().values
    lowest_low_14 = prices['low'].rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - prices['close'].values) / (highest_high_14 - lowest_low_14)
    
    # 1d volume confirmation: > 1.3x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: Williams %R crosses above -80 (oversold) with volume spike and 1w uptrend
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema_34_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: Williams %R crosses below -20 (overbought) with volume spike and 1w downtrend
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema_34_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when Williams %R returns to -50 midpoint
            if position == 1 and williams_r[i] >= -50:
                position = 0
                signals[i] = 0.0
            elif position == -1 and williams_r[i] <= -50:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals