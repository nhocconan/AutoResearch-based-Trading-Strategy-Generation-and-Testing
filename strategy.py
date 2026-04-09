#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R + 4h EMA50 trend filter + volume confirmation + session filter (08-20 UTC)
# - Primary signal: 1h Williams %R(14) < -80 (oversold) for long, > -20 (overbought) for short
# - Trend filter: 4h EMA50 - price must be above EMA for longs, below for shorts
# - Volume confirmation: 1h volume > 20-period median volume (avoid low-participation signals)
# - Session filter: Only trade between 08:00-20:00 UTC to avoid low-liquidity periods
# - Position size: 0.20 (discrete level) to minimize fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) per 1h strategy guidelines
# - Works in bull/bear: Williams %R captures mean reversion extremes, EMA50 filter ensures
#   trades align with higher timeframe trend, reducing false signals in strong trends

name = "1h_4h_williamsr_ema_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # Pre-compute 4h indicators
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 4h EMA50 for trend direction
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe (completed 4h bar only)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # 1h Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high_14 - lowest_low_14) == 0,
        -50.0,  # neutral when range is zero
        ((highest_high_14 - close) / (highest_high_14 - lowest_low_14)) * -100
    )
    
    # 1h volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(volume_regime[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R > -50 (exiting oversold) OR price crosses below EMA50
            if williams_r[i] > -50.0 or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: Williams %R < -50 (exiting overbought) OR price crosses above EMA50
            if williams_r[i] < -50.0 or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Williams %R extremes with volume confirmation and EMA50 filter
            # Long: Williams %R < -80 (oversold) AND volume regime AND price above EMA50
            if williams_r[i] < -80.0 and volume_regime[i] and close[i] > ema_50_aligned[i]:
                position = 1
                signals[i] = 0.20
            # Short: Williams %R > -20 (overbought) AND volume regime AND price below EMA50
            elif williams_r[i] > -20.0 and volume_regime[i] and close[i] < ema_50_aligned[i]:
                position = -1
                signals[i] = -0.20
    
    return signals