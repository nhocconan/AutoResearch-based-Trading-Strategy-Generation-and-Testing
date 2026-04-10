#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d trend filter and volume confirmation
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13) on 6h chart
# - Long when Bull Power > 0 AND Bear Power rising (less negative) AND volume > 1.3x 20-bar average AND 1d close > 1d EMA50
# - Short when Bear Power < 0 AND Bull Power falling (less positive) AND volume > 1.3x 20-bar average AND 1d close < 1d EMA50
# - Exit when Bull Power crosses below 0 (for longs) or Bear Power crosses above 0 (for shorts)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~15-25 trades/year (60-100 total over 4 years) to avoid fee drag
# - Elder Ray measures price strength relative to EMA, working well in both trending and ranging markets
# - Volume confirmation ensures participation, 1d trend filter aligns with higher timeframe momentum

name = "6h_1d_elder_ray_volume_trend_v1"
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
    
    # Pre-compute Elder Ray components on 6h data
    ema_period = 13
    close_6h = prices['close'].values
    ema_13 = pd.Series(close_6h).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    bull_power = high_6h - ema_13  # Bull Power = High - EMA
    bear_power = low_6h - ema_13   # Bear Power = Low - EMA
    
    # Pre-compute 12-period smoothed Elder Ray for trend confirmation (rising/falling)
    bull_power_smooth = pd.Series(bull_power).ewm(span=12, adjust=False, min_periods=12).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Pre-compute volume confirmation: > 1.3x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]) or
            np.isnan(volume_20_avg[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: Bull Power > 0 AND Bear Power rising (less negative) with volume spike and 1d uptrend
            if (bull_power[i] > 0 and 
                bear_power[i] > bear_power[i-1] and  # Bear Power rising (less negative)
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: Bear Power < 0 AND Bull Power falling (less positive) with volume spike and 1d downtrend
            elif (bear_power[i] < 0 and 
                  bull_power[i] < bull_power[i-1] and  # Bull Power falling (less positive)
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when Elder Ray crosses zero (mean reversion to EMA)
            if position == 1:
                if bull_power[i] <= 0:  # Bull Power crossed below zero
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                if bear_power[i] >= 0:  # Bear Power crossed above zero
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals