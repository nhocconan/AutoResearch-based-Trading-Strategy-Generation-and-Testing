#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation
# Uses 6h primary timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# 1d EMA50 provides trend filter to avoid counter-trend entries
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (13-period EMA on 6h)
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling (with volume confirmation)
# Short when Bear Power < 0 and falling, Bull Power > 0 and rising (with volume confirmation)
# Volume spike (>1.5 * 20-period EMA on 6h) confirms strong participation
# Discrete position sizing (0.25) minimizes fee churn while maintaining adequate exposure
# Works in bull (continuation) and bear (mean reversion via short) markets
# Designed to avoid overtrading by requiring confluence of momentum, trend, and volume

name = "6h_ElderRay_1dEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h data for Elder Ray calculation and volume confirmation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray (on 6h)
    close_6h = df_6h['close'].values
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = df_6h['high'].values - ema_13_6h  # High - EMA13
    bear_power = df_6h['low'].values - ema_13_6h   # Low - EMA13
    
    # Align Elder Ray components to 6h timeframe (data is already 6h, but using align for consistency)
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA (6h)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA50
        bullish_bias = close[i] > ema_50_1d_aligned[i]
        bearish_bias = close[i] < ema_50_1d_aligned[i]
        
        # Elder Ray momentum signals
        bull_rising = bull_power_aligned[i] > bull_power_aligned[i-1]
        bear_falling = bear_power_aligned[i] < bear_power_aligned[i-1]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias and bull_power_aligned[i] > 0 and bull_rising:
                # Long: bullish trend, positive bull power, rising bull power with volume spike
                if volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias and bear_power_aligned[i] < 0 and bear_falling:
                # Short: bearish trend, negative bear power, falling bear power with volume spike
                if volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop or counter-trend
        
        elif position == 1:  # Long position
            # Exit: bearish trend OR bear power turns positive OR bull power stops rising
            if (bearish_bias or bear_power_aligned[i] > 0 or not bull_rising):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: bullish trend OR bull power turns negative OR bear power stops falling
            if (bullish_bias or bull_power_aligned[i] < 0 or not bear_falling):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals