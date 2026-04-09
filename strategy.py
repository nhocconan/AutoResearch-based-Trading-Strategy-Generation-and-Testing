#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Williams %R with volume confirmation and ATR filter
# Williams %R(14) on 1w identifies overbought/oversold conditions on weekly timeframe
# Long when %R < -80 (oversold) with volume confirmation, short when %R > -20 (overbought)
# Volume confirmation: current 1d volume > 1.5x 20-period average filters low-quality signals
# ATR filter ensures sufficient volatility (avoid choppy low-vol periods)
# Fixed position size 0.25 to balance return and drawdown
# Target: 10-25 trades/year on 1d timeframe (40-100 total over 4 years)

name = "1d_1w_williamsr_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Williams %R (14-period)
    highest_high_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r_1w = -100 * (highest_high_1w - close_1w) / (highest_high_1w - lowest_low_1w)
    # Handle division by zero (when high == low)
    williams_r_1w = np.where((highest_high_1w - lowest_low_1w) == 0, -50, williams_r_1w)
    
    # Calculate 1d ATR (14-period) for volatility filtering
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF data to 1d timeframe
    williams_r_1w_aligned = align_htf_to_ltf(prices, df_1w, williams_r_1w)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1d)
    
    # Pre-compute volume confirmation (20-period average for 1d)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC) - though less relevant for 1d, keep for consistency
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(williams_r_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or not in_session[i] or
            atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average 1d volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility filter: only trade when 1d ATR is above its 50-period average
        atr_ma_50_1d = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=50).mean()
        if len(atr_ma_50_1d) > i:
            vol_filter = atr_1d_aligned[i] > atr_ma_50_1d.iloc[i]
        else:
            vol_filter = True  # Not enough data for MA, allow trading
            
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit when Williams %R returns above -50 (mean reversion) or stop at 2*ATR
            if williams_r_1w_aligned[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit when Williams %R returns below -50 (mean reversion) or stop at 2*ATR
            if williams_r_1w_aligned[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Williams %R trading with volume and volatility confirmation
            # Long when oversold (%R < -80), short when overbought (%R > -20)
            if volume_confirmed:
                if williams_r_1w_aligned[i] < -80:  # Oversold - go long
                    position = 1
                    signals[i] = position_size
                elif williams_r_1w_aligned[i] > -20:  # Overbought - go short
                    position = -1
                    signals[i] = -position_size
    
    return signals