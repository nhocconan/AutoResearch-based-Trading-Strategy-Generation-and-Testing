#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d volume filter and 4h trend alignment
# - Williams %R(14) identifies overbought/oversold conditions on 12h chart
# - Long when %R < -80 (oversold) AND volume > 1.3x 20-bar average AND 4h close > 4h EMA20
# - Short when %R > -20 (overbought) AND volume > 1.3x 20-bar average AND 4h close < 4h EMA20
# - Exit when %R returns to -50 (mean reversion center) or opposite extreme is reached
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~15-25 trades/year (60-100 total over 4 years) to avoid fee drag
# - Williams %R works well in ranging markets (2022-2025) and catches reversals in trends
# - Volume confirmation filters false signals
# - 4h trend filter ensures we trade with higher timeframe momentum

name = "12h_4h_1d_williamsr_meanreversion_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Williams %R(14) on 12h data
    period = 14
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Calculate highest high and lowest low over the period
    highest_high = pd.Series(high_12h).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low_12h).rolling(window=period, min_periods=period).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Values range from -100 (oversold) to 0 (overbought)
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_12h) / (highest_high - lowest_low)) * -100,
        -50  # Default when range is zero
    )
    
    # Pre-compute 12h volume confirmation: > 1.3x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    # Pre-compute 4h EMA(20) for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(ema_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: Williams %R < -80 (oversold) with volume spike and 4h uptrend
            if (williams_r[i] < -80 and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema_20_4h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: Williams %R > -20 (overbought) with volume spike and 4h downtrend
            elif (williams_r[i] > -20 and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema_20_4h_aligned[i]):
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