#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h VWAP deviation and 1d ATR-based volatility filter.
# Long when price closes above 12h VWAP + 1.5*ATR(1d) with expanding volume and bullish 1d EMA(50).
# Short when price closes below 12h VWAP - 1.5*ATR(1d) with expanding volume and bearish 1d EMA(50).
# Exit when price crosses back to 12h VWAP or volatility contracts (ATR ratio < 0.8).
# Designed to capture volatility expansion moves in both bull and bear markets with strict entry filters.
# Target: 20-30 trades/year per symbol (80-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for VWAP
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate VWAP for 12h
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    vwap_num = np.cumsum(typical_price_12h * volume_12h)
    vwap_den = np.cumsum(volume_12h)
    vwap_12h = vwap_num / vwap_den
    vwap_12h[vwap_den == 0] = np.nan
    
    # Load 1d data ONCE for ATR and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) on 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_period = 14
    atr_1d = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate EMA(50) on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to lower timeframe
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20, 50)  # Need VWAP, volume MA, and EMA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_12h_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: EMA(50) slope
        ema_slope = ema_50_1d_aligned[i] - ema_50_1d_aligned[i-1] if i > 0 else 0
        uptrend = ema_slope > 0
        downtrend = ema_slope < 0
        
        if position == 0:
            # Look for VWAP breakouts with volatility expansion
            # Long: price close above VWAP + 1.5*ATR AND uptrend AND volume confirmation
            if (close[i] > vwap_12h_aligned[i] + 1.5 * atr_1d_aligned[i] and 
                uptrend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price close below VWAP - 1.5*ATR AND downtrend AND volume confirmation
            elif (close[i] < vwap_12h_aligned[i] - 1.5 * atr_1d_aligned[i] and 
                  downtrend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back to VWAP or volatility contracts
            if (close[i] <= vwap_12h_aligned[i] or 
                atr_1d_aligned[i] < 0.8 * atr_1d_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back to VWAP or volatility contracts
            if (close[i] >= vwap_12h_aligned[i] or 
                atr_1d_aligned[i] < 0.8 * atr_1d_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_VWAP_ATR_VolumeTrendFilter_v1"
timeframe = "4h"
leverage = 1.0