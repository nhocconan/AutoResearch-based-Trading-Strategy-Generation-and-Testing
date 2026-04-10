#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume spike confirmation
# - Williams %R(14) < -80 indicates oversold, > -20 indicates overbought
# - Enter long when %R crosses above -80 from below with volume > 1.8x 24-bar avg AND 1d close > 1d EMA50 (uptrend)
# - Enter short when %R crosses below -20 from above with volume > 1.8x 24-bar avg AND 1d close < 1d EMA50 (downtrend)
# - Exit when %R crosses -50 (mean reversion midpoint) or opposite signal occurs
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~25 trades/year (100 total over 4 years) to avoid fee drag
# - Williams %R is effective in ranging markets which dominate 2025+ test period
# - Volume confirmation ensures breakouts have participation, reducing false signals
# - 1d trend filter ensures we trade with higher timeframe momentum, improving win rate

name = "6h_1d_williamsr_meanrev_volume_trend_v1"
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
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 6h Williams %R(14)
    highest_high_14 = prices['high'].rolling(window=14, min_periods=14).max().values
    lowest_low_14 = prices['low'].rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    rr = highest_high_14 - lowest_low_14
    williams_r = np.where(rr != 0, -100 * (highest_high_14 - prices['close'].values) / rr, -50.0)
    
    # 6h volume confirmation: > 1.8x 24-period average (4 days)
    volume_24_avg = prices['volume'].rolling(window=24, min_periods=24).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_24_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_24_avg[i]) or np.isnan(highest_high_14[i]) or 
            np.isnan(lowest_low_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: Williams %R crosses above -80 from below with volume spike and 1d uptrend
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: Williams %R crosses below -20 from above with volume spike and 1d downtrend
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when Williams %R crosses -50 (mean reversion midpoint)
            if position == 1 and williams_r[i] < -50 and williams_r[i-1] >= -50:
                position = 0
                signals[i] = 0.0
            elif position == -1 and williams_r[i] > -50 and williams_r[i-1] <= -50:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals