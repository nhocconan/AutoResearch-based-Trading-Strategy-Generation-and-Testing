#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Williams Alligator: Jaw (EMA13 smoothed 8), Teeth (EMA8 smoothed 5), Lips (EMA5 smoothed 3)
# Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish)
# 1w EMA50 filter ensures alignment with weekly trend to avoid counter-trend whipsaws
# Volume spike (>1.8 * 20-period EMA) confirms institutional participation
# Designed for low trade frequency: ~12-25 trades/year per symbol with 0.25 sizing
# Works in bull markets via trend continuation and bear markets via trend-following alignment
# BTC/ETH focused: requires weekly trend alignment and volume spike to avoid SOL-only bias

name = "12h_WilliamsAlligator_1wEMA50_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator components (all calculated on 12h data)
    # Jaw: Blue line - EMA13 smoothed by 8 periods
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = pd.Series(ema13).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Teeth: Red line - EMA8 smoothed by 5 periods
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = pd.Series(ema8).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Lips: Green line - EMA5 smoothed by 3 periods
    ema5 = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = pd.Series(ema5).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Volume confirmation: volume > 1.8 * 20-period EMA (strict filter)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for Alligator (13+8+5=26, plus buffers)
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA50
        bullish_bias = close[i] > ema_50_1w_aligned[i]
        bearish_bias = close[i] < ema_50_1w_aligned[i]
        
        # Williams Alligator signals
        bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias and bullish_alignment:
                # Long: Alligator bullish alignment with weekly trend and volume spike
                if volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias and bearish_alignment:
                # Short: Alligator bearish alignment with weekly trend and volume spike
                if volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # No clear signal
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish OR price breaks below weekly EMA50
            if not bullish_alignment or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish OR price breaks above weekly EMA50
            if not bearish_alignment or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals