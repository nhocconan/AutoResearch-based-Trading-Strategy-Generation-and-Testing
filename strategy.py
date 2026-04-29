#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout with 12h Trend and Volume Spike
# Camarilla pivot levels (R3/S3) act as strong intraday support/resistance
# Breakout above R3 or below S3 with volume confirmation indicates strong momentum
# 12h EMA50 trend filter ensures we trade breakouts in direction of higher timeframe trend
# Works in both bull and bear markets by capturing volatility expansion after consolidation
# Target: 20-50 trades/year (80-200 total over 4 years)

name = "4h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Range = high - low
    price_range = high - low
    
    # Camarilla levels (based on previous day's action)
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    camarilla_r3 = close + (price_range * 1.1 / 4.0)
    camarilla_s3 = close - (price_range * 1.1 / 4.0)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 20)  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema50_12h = ema50_12h_aligned[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        
        # Determine trend regime from 12h EMA50
        bullish_regime = curr_close > curr_ema50_12h
        bearish_regime = curr_close < curr_ema50_12h
        
        if position == 0:  # Flat - look for new entries
            # Look for breakout with volume confirmation
            if curr_volume_confirm:
                # Bullish breakout: price breaks above R3 in bullish regime
                if bullish_regime and curr_close > curr_r3:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below S3 in bearish regime
                elif bearish_regime and curr_close < curr_s3:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price crosses below R3 (failed breakout) OR trend reverses
            if curr_close < curr_r3 or curr_close < curr_ema50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price crosses above S3 (failed breakout) OR trend reverses
            if curr_close > curr_s3 or curr_close > curr_ema50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals