#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels (R3/S3) act as strong intraday support/resistance where breakouts often continue.
# 1d EMA34 provides higher timeframe trend bias to avoid counter-trend trades.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 20-50 trades/year to minimize fee drag while maintaining edge in both bull and bear markets.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # 1d HTF data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (H + L + C) / 3
    typical_price = (high + low + close) / 3.0
    # Use previous day's typical price for Camarilla calculation
    prev_typical = pd.Series(typical_price).shift(1).values
    
    # Camarilla levels: R3 = PP + (H-L)*1.1/2, S3 = PP - (H-L)*1.1/2
    # Where PP = (H+L+C)/3 from previous day
    high_prev = pd.Series(high).shift(1).values
    low_prev = pd.Series(low).shift(1).values
    close_prev = pd.Series(close).shift(1).values
    
    pp = (high_prev + low_prev + close_prev) / 3.0
    rang = high_prev - low_prev
    r3 = pp + (rang * 1.1 / 2)
    s3 = pp - (rang * 1.1 / 2)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need sufficient history for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Trend filter: price above/below 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Camarilla breakout conditions
        breakout_up = curr_close > r3[i]  # Break above R3 resistance
        breakout_down = curr_close < s3[i]  # Break below S3 support
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3, volume spike, uptrend
            if breakout_up and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3, volume spike, downtrend
            elif breakout_down and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on break below S3 or trend reversal
            if curr_close < s3[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on break above R3 or trend reversal
            if curr_close > r3[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals