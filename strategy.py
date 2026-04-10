#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1d volume confirmation + 1w trend filter
# - Primary: 6h Williams %R(14) < -80 for long, > -20 for short (extreme oversold/overbought)
# - Volume filter: 1d volume > 1.5x 20-period volume MA to confirm participation
# - Trend filter: 1w close > 1w EMA(34) for long bias, < for short bias (only trade with weekly trend)
# - Exit: Williams %R returns to -50 level (mean reversion)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Williams %R captures reversals, volume confirms, weekly trend filters counter-trend
# - Target: 75-150 total trades over 4 years = 19-37/year for 6h timeframe

name = "6h_1d_1w_williamsr_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    close_1w = df_1w['close'].values
    
    # Calculate 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Calculate 1d volume confirmation: volume > 1.5x 20-period volume MA
    volume_ma_20_1d = pd.Series(volume_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Calculate 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align HTF indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)  # Williams %R is already 6h, but we need to align the calculation base
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Also need 1d volume for confirmation
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period MA
        vol_confirm = volume_1d_aligned[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) + vol confirmation + weekly uptrend
            if williams_r_aligned[i] < -80 and vol_confirm and close[i] > ema_34_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R > -20 (overbought) + vol confirmation + weekly downtrend
            elif williams_r_aligned[i] > -20 and vol_confirm and close[i] < ema_34_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit at Williams %R = -50 (mean reversion)
            # Exit: Williams %R returns to -50 level
            if position == 1:  # Long position
                if williams_r_aligned[i] >= -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if williams_r_aligned[i] <= -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals