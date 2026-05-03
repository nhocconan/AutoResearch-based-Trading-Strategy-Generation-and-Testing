#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot (R3/S3) breakout with 1d EMA50 trend filter and volume confirmation.
# Uses 4h for signal direction (Camarilla breakout + trend) and 1h for precise entry timing.
# Session filter (08-20 UTC) reduces noise trades. Discrete sizing 0.20 to manage fees.
# Target: 15-37 trades/year per symbol to avoid fee drag. Works in bull/bear via trend filter.

name = "1h_Camarilla_R3S3_4hTrend_1dEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:  # Need sufficient data for Camarilla calculation
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivots (based on prior completed 4h bar)
    # Camarilla levels: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #                 S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # Using prior completed 4h bar's OHLC
    prior_close_4h = np.roll(df_4h['close'].values, 1)
    prior_high_4h = np.roll(df_4h['high'].values, 1)
    prior_low_4h = np.roll(df_4h['low'].values, 1)
    prior_close_4h[0] = np.nan
    prior_high_4h[0] = np.nan
    prior_low_4h[0] = np.nan
    
    # Calculate Camarilla R3 and S3 for prior 4h bar
    camarilla_r3 = prior_close_4h + 1.1 * (prior_high_4h - prior_low_4h)
    camarilla_s3 = prior_close_4h - 1.1 * (prior_high_4h - prior_low_4h)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Calculate 4h trend filter: EMA50 on 4h close
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data for EMA50 trend filter (additional confirmation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 24-bar average (on 1h data)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient warmup
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0  # Force flat outside session
            position = 0
            continue
            
        # Get current values
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        ema_trend_4h = ema_50_4h_aligned[i]
        ema_trend_1d = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(r3) or np.isnan(s3) or np.isnan(ema_trend_4h) or np.isnan(ema_trend_1d):
            continue
            
        # Entry conditions
        # Long: break above Camarilla R3 with volume spike and above both EMAs (bullish alignment)
        long_entry = (close[i] > r3) and vol_spike and (close[i] > ema_trend_4h) and (close[i] > ema_trend_1d)
        # Short: break below Camarilla S3 with volume spike and below both EMAs (bearish alignment)
        short_entry = (close[i] < s3) and vol_spike and (close[i] < ema_trend_4h) and (close[i] < ema_trend_1d)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 (mean reversion) or trend turns bearish
            if close[i] < s3 or close[i] < ema_trend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 (mean reversion) or trend turns bullish
            if close[i] > r3 or close[i] > ema_trend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals