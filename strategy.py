#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d/1w Williams %R extremes with volume confirmation and ATR filter
# Williams %R(14) < -80 = oversold (long), > -20 = overbought (short) on 1d
# Weekly trend filter: only trade long when price > weekly EMA(50), short when price < weekly EMA(50)
# Volume confirmation: current 6h volume > 1.5x 20-period average filters low-quality signals
# ATR filter ensures sufficient volatility (avoid choppy low-vol periods)
# Position size fixed at 0.25 to balance return and drawdown
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_1w_williamsr_atr_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Williams %R (14-period)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r[highest_high_14 == lowest_low_14] = -50  # Avoid division by zero
    
    # Calculate 1w EMA (50-period) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d ATR (14-period) for volatility filtering
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or not in_session[i] or
            atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume
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
            # Exit on Williams %R > -20 (overbought) or stop at close < weekly EMA
            if williams_r_aligned[i] > -20:
                position = 0
                signals[i] = 0.0
            elif close[i] < ema_50_1w_aligned[i]:  # Stop loss: close below weekly EMA
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on Williams %R < -80 (oversold) or stop at close > weekly EMA
            if williams_r_aligned[i] < -80:
                position = 0
                signals[i] = 0.0
            elif close[i] > ema_50_1w_aligned[i]:  # Stop loss: close above weekly EMA
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Williams %R extreme + volume confirmation + weekly trend filter
            if volume_confirmed:
                # Long: Williams %R < -80 (oversold) AND price > weekly EMA (uptrend)
                if williams_r_aligned[i] < -80 and close[i] > ema_50_1w_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short: Williams %R > -20 (overbought) AND price < weekly EMA (downtrend)
                elif williams_r_aligned[i] > -20 and close[i] < ema_50_1w_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals