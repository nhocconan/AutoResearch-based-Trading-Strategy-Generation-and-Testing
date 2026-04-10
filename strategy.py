#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume confirmation
# - Primary: 6h Williams %R(14) < -80 for long, > -20 for short (extreme oversold/overbought)
# - HTF trend: 1w EMA(21) slope determines market regime (rising = long bias, falling = short bias)
# - HTF volume: 1w volume > 1.5x 20-period MA for institutional participation
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Long: Williams %R < -80 + rising 1w EMA slope + volume spike + session
# - Short: Williams %R > -20 + falling 1w EMA slope + volume spike + session
# - Exit: Williams %R crosses -50 (mean reversion complete) or EMA slope reverses
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# - Works in bull/bear: Williams %R captures mean reversion in ranges, EMA slope filters trend strength, volume confirms participation

name = "6h_1w_williamsr_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Calculate 1w EMA(21) and its slope
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate EMA slope (rate of change over 3 periods)
    ema_slope = np.zeros_like(ema_1w_aligned)
    for i in range(3, len(ema_1w_aligned)):
        if not np.isnan(ema_1w_aligned[i]) and not np.isnan(ema_1w_aligned[i-3]):
            ema_slope[i] = (ema_1w_aligned[i] - ema_1w_aligned[i-3]) / 3
        else:
            ema_slope[i] = np.nan
    
    # Calculate 1w volume MA(20)
    volume_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(williams_r[i]) or np.isnan(ema_1w_aligned[i]) or np.isnan(ema_slope[i]) or
            np.isnan(volume_ma_20_1w_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w volume > 1.5x 20-period MA
        volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
        volume_confirm = volume_1w_aligned[i] > 1.5 * volume_ma_20_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 + rising 1w EMA slope + volume spike + session
            if (williams_r[i] < -80 and ema_slope[i] > 0 and volume_confirm):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R > -20 + falling 1w EMA slope + volume spike + session
            elif (williams_r[i] > -20 and ema_slope[i] < 0 and volume_confirm):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Williams %R crosses -50 (mean reversion complete) or EMA slope reverses
            if position == 1:  # Long position
                if williams_r[i] > -50 or ema_slope[i] < 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if williams_r[i] < -50 or ema_slope[i] > 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals