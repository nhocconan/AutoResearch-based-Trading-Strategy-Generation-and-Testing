#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter and ATR-based volatility regime
# Uses tighter volume confirmation (1.5x EMA) and discrete sizing (0.25) to reduce trade frequency.
# Designed for 75-200 total trades over 4 years (19-50/year) with focus on BTC/ETH performance.
# Camarilla levels provide structured support/resistance; 1d EMA50 ensures trend alignment.
# ATR regime filter avoids choppy markets where breakouts fail.

name = "4h_Camarilla_R3S3_1dEMA50_ATRVol_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA50 trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) on 4h for volatility regime filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Volatility regime: ATR ratio < 1.2 (low volatility) for breakout reliability
    vol_regime = atr_14 < (1.2 * atr_ma_50)
    
    # Calculate Camarilla levels from previous 1d bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    R3 = prev_close + 1.1 * (prev_high - prev_low)
    S3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: 20-period EMA on 4h
    vol_ema_20 = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema_20_values = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20[:] = vol_ema_20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN or outside session or unfavorable volatility regime
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(vol_ema_20[i]) or not in_session[i] or not vol_regime[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Tighter volume confirmation: 1.5x EMA (reduces false signals)
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above R3 in uptrend alignment with volume spike
            if close[i] > R3_aligned[i] and ema_50_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 in downtrend alignment with volume spike
            elif close[i] < S3_aligned[i] and ema_50_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or loses uptrend alignment
            if close[i] < S3_aligned[i] or ema_50_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 or loses downtrend alignment
            if close[i] > R3_aligned[i] or ema_50_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals