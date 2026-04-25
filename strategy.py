#!/usr/bin/env python3
"""
4h Camarilla Pivot R3/S3 Breakout + 1d EMA34 Trend + Volume Spike + ATR Stop
Hypothesis: Camarilla R3/S3 levels act as strong intraday support/resistance. 
Breakouts with volume spike and 1d EMA34 trend filter capture momentum moves.
ATR-based stoploss manages risk. Works in bull/bear by trend-filtering.
Target: 20-50 trades/year (80-200 over 4 years).
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get previous day's high, low, close for Camarilla levels
    # Since we're on 4h timeframe, we need daily OHLC
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    high_1d = df_1d_ohlc['high'].values
    low_1d = df_1d_ohlc['low'].values
    close_1d = df_1d_ohlc['close'].values
    
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_r3, additional_delay_bars=1)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_s3, additional_delay_bars=1)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA34 warmup and ATR
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        atr_val = atr[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Update trailing stop for existing positions
        if position == 1:
            # Long: trail stop below highest high since entry
            if curr_close <= entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short: trail stop above lowest low since entry
            if curr_close >= entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
        
        # Look for new entries only when flat
        if position == 0:
            # Long: break above R3 with volume spike and above 1d EMA34 (uptrend)
            long_condition = (curr_close > r3_level) and volume_spike and (curr_close > ema_trend)
            # Short: break below S3 with volume spike and below 1d EMA34 (downtrend)
            short_condition = (curr_close < s3_level) and volume_spike and (curr_close < ema_trend)
            
            if long_condition:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0