#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Primary: 4h price breaks above/below Camarilla H3/L3 levels from prior 1d
# - HTF trend: 1d EMA(50) slope determines bias (rising = long bias, falling = short bias)
# - Volume confirmation: 4h volume > 1.5x 20-period MA
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Long: price > H3 + rising 1d EMA slope + volume spike + session
# - Short: price < L3 + falling 1d EMA slope + volume spike + session
# - Exit: price reverts to Camarilla Pivot point or EMA slope reverses
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# - Works in bull/bear: Camarilla levels provide structure, EMA slope filters trend strength

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h volume MA(20)
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA(50) and its slope
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate EMA slope (rate of change over 3 periods)
    ema_slope = np.zeros_like(ema_1d_aligned)
    for i in range(3, len(ema_1d_aligned)):
        if not np.isnan(ema_1d_aligned[i]) and not np.isnan(ema_1d_aligned[i-3]):
            ema_slope[i] = (ema_1d_aligned[i] - ema_1d_aligned[i-3]) / 3
        else:
            ema_slope[i] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(60, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(ema_slope[i]) or
            np.isnan(volume_ma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels from prior 1d candle (use aligned data)
        # Get prior completed 1d bar data
        prior_high = high_1d_aligned[i-1] if i > 0 else np.nan
        prior_low = low_1d_aligned[i-1] if i > 0 else np.nan
        prior_close = close_1d_aligned[i-1] if i > 0 else np.nan
        
        if np.isnan(prior_high) or np.isnan(prior_low) or np.isnan(prior_close):
            signals[i] = 0.0
            continue
            
        # Camarilla levels
        pivot = (prior_high + prior_low + prior_close) / 3
        range_ = prior_high - prior_low
        h3 = pivot + (range_ * 1.1 / 4)
        l3 = pivot - (range_ * 1.1 / 4)
        h4 = pivot + (range_ * 1.1 / 2)
        l4 = pivot - (range_ * 1.1 / 2)
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * volume_ma_20[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > H3 + rising 1d EMA slope + volume spike + session
            if (close[i] > h3 and ema_slope[i] > 0 and volume_confirm):
                position = 1
                signals[i] = 0.25
            # Short entry: price < L3 + falling 1d EMA slope + volume spike + session
            elif (close[i] < l3 and ema_slope[i] < 0 and volume_confirm):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: price reverts to pivot or EMA slope reverses
            if position == 1:  # Long position
                if close[i] < pivot or ema_slope[i] < 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] > pivot or ema_slope[i] > 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals