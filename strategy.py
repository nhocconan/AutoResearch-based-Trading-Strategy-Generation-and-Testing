#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d trend filter and volume confirmation
# - Primary: Williams %R(14) < -80 for long, > -20 for short (oversold/overbought)
# - Trend filter: 1d close > EMA50 for long bias, < EMA50 for short bias
# - Volume confirmation: 4h volume > 1.5x 20-period volume MA
# - Exit: Williams %R crosses back above -50 (long) or below -50 (short)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Williams %R captures mean reversion in ranging markets,
#   1d EMA50 filter ensures we trade with higher timeframe trend
# - Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe

name = "4h_1d_williams_r_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 for 1d trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams %R(14) for 4h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    
    williams_r = -100 * ((highest_high - close) / hl_range)
    
    # Calculate 4h volume confirmation: volume > 1.5x 20-period volume MA
    volume_ma_20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * volume_ma_20[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) + above 1d EMA50 + vol confirmation
            if (williams_r[i] < -80 and 
                close[i] > ema50_1d_aligned[i] and vol_confirm):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R > -20 (overbought) + below 1d EMA50 + vol confirmation
            elif (williams_r[i] > -20 and 
                  close[i] < ema50_1d_aligned[i] and vol_confirm):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when Williams %R crosses -50
            # Exit: Williams %R crosses back above -50 (long) or below -50 (short)
            if position == 1:  # Long position
                if williams_r[i] >= -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if williams_r[i] <= -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals