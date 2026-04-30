#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 12h HMA21 trend filter and volume confirmation
# Camarilla R4/S4 levels represent stronger intraday support/resistance than R3/S3
# 12h HMA21 filter ensures alignment with higher timeframe trend (reduces counter-trend trades)
# Volume spike (1.8x 24-period average) confirms institutional participation
# Works in bull markets via breakouts above R4 and bear markets via breakdowns below S4
# Discrete sizing 0.25 minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Camarilla_R4_S4_Breakout_12hHMA21_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 12h data ONCE before loop (MTF Rule #1)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h HMA21 (Hull Moving Average)
    close_12h = df_12h['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='same')
    
    wma_half = wma(close_12h, half_len)
    wma_full = wma(close_12h, 21)
    raw_hma = 2 * wma_half - wma_full
    hma_21_12h = wma(raw_hma, sqrt_len)
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Calculate 12h Camarilla pivot levels (R4, S4)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    daily_range = high_12h - low_12h
    camarilla_r4 = close_12h + daily_range * 1.1 / 2
    camarilla_s4 = close_12h - daily_range * 1.1 / 2
    
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Volume confirmation: volume > 1.8x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 21, 24)  # warmup for HMA21, Camarilla, and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(hma_21_12h_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_hma_21_12h = hma_21_12h_aligned[i]
        curr_camarilla_r4 = camarilla_r4_aligned[i]
        curr_camarilla_s4 = camarilla_s4_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: break above Camarilla R4 AND above 12h HMA21 (uptrend)
                if curr_high > curr_camarilla_r4 and curr_close > curr_hma_21_12h:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below Camarilla S4 AND below 12h HMA21 (downtrend)
                elif curr_low < curr_camarilla_s4 and curr_close < curr_hma_21_12h:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below Camarilla S4 (breakout fails)
            if curr_close < curr_camarilla_s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above Camarilla R4 (breakdown fails)
            if curr_close > curr_camarilla_r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals