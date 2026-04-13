#!/usr/bin/env python3
"""
1h_4h_1d_Volume_Regime_Filter
Hypothesis: Trade volatility regime shifts using 4h trend and 1d volume regime. 
Long when 4h EMA21 rising + 1d volume > 1.5x 20-day avg + price > 4h VWAP. 
Short when 4h EMA21 falling + 1d volume > 1.5x 20-day avg + price < 4h VWAP. 
Enter on 1h break of 4h VWAP with volume confirmation. 
Uses volume regime to filter false breaks. Target: 15-37 trades/year.
Works in bull (trend up + volume) and bear (trend down + volume).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend and VWAP
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 4h EMA21 for trend
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_rising = ema_21_4h > np.roll(ema_21_4h, 1)
    ema_falling = ema_21_4h < np.roll(ema_21_4h, 1)
    ema_rising[0] = False
    ema_falling[0] = False
    
    # 4h VWAP (session VWAP reset daily)
    vwap_4h = np.zeros_like(close_4h)
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    cum_vol = 0.0
    cum_pv = 0.0
    for i in range(len(close_4h)):
        if i > 0 and df_4h.index[i].date() != df_4h.index[i-1].date():
            cum_vol = 0.0
            cum_pv = 0.0
        cum_vol += volume_4h[i]
        cum_pv += typical_price_4h[i] * volume_4h[i]
        vwap_4h[i] = cum_pv / cum_vol if cum_vol > 0 else typical_price_4h[i]
    
    # Get 1d data for volume regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_regime = volume_1d > (vol_ma_20_1d * 1.5)
    
    # Align all to 1h
    ema_rising_aligned = align_htf_to_ltf(prices, df_4h, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_4h, ema_falling)
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20
    
    for i in range(50, n):
        if not session_mask[i] or \
           np.isnan(ema_rising_aligned[i]) or \
           np.isnan(ema_falling_aligned[i]) or \
           np.isnan(vwap_4h_aligned[i]) or \
           np.isnan(volume_regime_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation on 1h
        vol_ma_20_1h = np.nan
        if i >= 20:
            vol_ma_20_1h = np.mean(volume[max(0, i-19):i+1])
        volume_expansion_1h = volume[i] > (vol_ma_20_1h * 1.5) if not np.isnan(vol_ma_20_1h) else False
        
        # Long: 4h uptrend + volume regime + price > VWAP
        if ema_rising_aligned[i] and volume_regime_aligned[i] and close[i] > vwap_4h_aligned[i]:
            if volume_expansion_1h and close[i] > close[i-1]:
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            elif position == 1:
                signals[i] = position_size
            else:
                signals[i] = 0.0
        # Short: 4h downtrend + volume regime + price < VWAP
        elif ema_falling_aligned[i] and volume_regime_aligned[i] and close[i] < vwap_4h_aligned[i]:
            if volume_expansion_1h and close[i] < close[i-1]:
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        else:
            # Exit conditions
            if position == 1 and (close[i] < vwap_4h_aligned[i] or not volume_regime_aligned[i]):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (close[i] > vwap_4h_aligned[i] or not volume_regime_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_Volume_Regime_Filter"
timeframe = "1h"
leverage = 1.0