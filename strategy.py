#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d volume spike + 1w trend filter
# - Primary: 6h Williams %R(14) for mean reversion signals (<20 oversold, >80 overbought)
# - HTF: 1d volume confirmation (current volume > 2.0x 20-period MA) for conviction
# - HTF: 1w EMA(21) trend filter (only long when price > EMA, only short when price < EMA)
# - Session: Only trade during 08-20 UTC to avoid low-liquidity hours
# - Long: Williams %R < 20 + volume confirmation + price > weekly EMA + session active
# - Short: Williams %R > 80 + volume confirmation + price < weekly EMA + session active
# - Exit: Williams %R crosses back above 50 (for long) or below 50 (for short) OR session ends
# - Position sizing: 0.25 (discrete level to balance return and drawdown)
# - Works in bull/bear: Williams %R captures reversals, volume confirms breakouts, weekly EMA filters counter-trend trades
# - Target: 50-120 total trades over 4 years (12-30/year) for 6h timeframe

name = "6h_1d_1w_williamsr_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 25 or len(df_1w) < 5:  # Need enough data for indicators
        return np.zeros(n)
    
    # Pre-compute 6h data
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    
    # Pre-compute 1d data for volume confirmation
    volume_1d = df_1d['volume'].values
    
    # Pre-compute 1w data for trend filter
    close_1w = df_1w['close'].values
    
    # Calculate 6h Williams %R(14) - using min_periods
    highest_high_14 = np.full(len(high_6h), np.nan)
    lowest_low_14 = np.full(len(low_6h), np.nan)
    williams_r = np.full(len(close_6h), np.nan)
    
    for i in range(13, len(high_6h)):  # Start from index 13 for 14-period lookback
        if not (np.isnan(high_6h[i-13:i+1]).any() or np.isnan(low_6h[i-13:i+1]).any()):
            highest_high_14[i] = np.max(high_6h[i-13:i+1])
            lowest_low_14[i] = np.min(low_6h[i-13:i+1])
            if highest_high_14[i] != lowest_low_14[i]:
                williams_r[i] = (highest_high_14[i] - close_6h[i]) / (highest_high_14[i] - lowest_low_14[i]) * -100
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        if not np.isnan(volume_1d[i-19:i+1]).any():
            volume_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Calculate 1w EMA(21) for trend filter
    if len(close_1w) >= 21:
        close_1w_series = pd.Series(close_1w)
        ema_21_1w = close_1w_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    else:
        ema_21_1w = np.full(len(close_1w), np.nan)
    
    # Align all HTF indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)  # Williams %R is already 6h, but we use 1d for alignment reference
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Align 1d volume for direct comparison
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_active = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from second bar to avoid index issues
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period MA
        volume_confirm = volume_1d_aligned[i] > 2.0 * volume_ma_20_1d_aligned[i]
        
        # Session filter: only trade during 08-20 UTC
        in_session = session_active[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < 20 (oversold) + volume confirmation + price > weekly EMA + session active
            if williams_r_aligned[i] < 20 and volume_confirm and close_6h[i] > ema_21_1w_aligned[i] and in_session:
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R > 80 (overbought) + volume confirmation + price < weekly EMA + session active
            elif williams_r_aligned[i] > 80 and volume_confirm and close_6h[i] < ema_21_1w_aligned[i] and in_session:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Williams %R crosses back above 50 (for long) or below 50 (for short) OR session ends
            if position == 1:  # Long position
                if williams_r_aligned[i] > 50 or not in_session:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if williams_r_aligned[i] < 50 or not in_session:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals