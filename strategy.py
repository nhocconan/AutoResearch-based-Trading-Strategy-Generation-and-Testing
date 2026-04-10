#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h volume filter and 1d trend alignment
# - Williams %R(14) identifies overbought/oversold conditions on 6h chart
# - Long when %R < -80 (oversold) AND volume > 1.5x 20-bar average AND 1d close > 1d EMA50
# - Short when %R > -20 (overbought) AND volume > 1.5x 20-bar average AND 1d close < 1d EMA50
# - Exit when %R returns to -50 (mean reversion center) or opposite extreme is reached
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~20-30 trades/year (80-120 total over 4 years) to avoid fee drag
# - Williams %R works well in ranging markets (2022-2025) and catches reversals in trends
# - Volume confirmation filters false signals
# - 1d trend filter ensures we trade with higher timeframe momentum

name = "6h_12h_1d_williamsr_meanreversion_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Williams %R(14) on 6h data
    period = 14
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Calculate highest high and lowest low over the period
    highest_high = pd.Series(high_6h).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low_6h).rolling(window=period, min_periods=period).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Values range from -100 (oversold) to 0 (overbought)
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_6h) / (highest_high - lowest_low)) * -100,
        -50  # Default when range is zero
    )
    
    # Pre-compute 12h volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: Williams %R < -80 (oversold) with volume spike and 1d uptrend
            if (williams_r[i] < -80 and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: Williams %R > -20 (overbought) with volume spike and 1d downtrend
            elif (williams_r[i] > -20 and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Williams %R returns to -50 (mean reversion center)
            # 2. Opposite extreme is reached (long exits at > -20, short exits at < -80)
            if position == 1:
                if williams_r[i] >= -50 or williams_r[i] > -20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                if williams_r[i] <= -50 or williams_r[i] < -80:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals