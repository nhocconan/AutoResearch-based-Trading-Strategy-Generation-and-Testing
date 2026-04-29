#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume spike and 1d trend filter
# Long when price breaks above R3 level with 4h volume > 2x 20-period average and 1d close > 1d EMA50
# Short when price breaks below S3 level with 4h volume > 2x 20-period average and 1d close < 1d EMA50
# Uses 1d EMA50 for trend filter to avoid counter-trend trades, 4h volume for confirmation
# Camarilla levels provide precise intraday support/resistance
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag on 1h timeframe
# Session filter: 08-20 UTC to avoid low-volume Asian session noise

name = "1h_Camarilla_R3S3_4hVolumeSpike_1dTrend_Filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (precompute hour array)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h volume MA(20)
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for EMA and indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if HTF data not available
        if np.isnan(vol_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels for today using previous day's OHLC
        # Need to get previous day's OHLC - we'll approximate using rolling window
        # For 1h timeframe, we need to look back 24 hours for previous day
        if i < 24:
            continue
            
        # Get previous day's OHLC (24 hours ago to 1 hour ago)
        prev_high = np.max(high[i-24:i])
        prev_low = np.min(low[i-24:i])
        prev_close = close[i-1]
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla R3 and S3 levels
        r3 = prev_close + (range_val * 1.1 / 4)
        s3 = prev_close - (range_val * 1.1 / 4)
        
        # Volume confirmation: 4h volume > 2x 20-period average
        volume_confirm = volume[i] > (2.0 * vol_4h_aligned[i])
        
        # Trend filter: 1d close > EMA50 for long, < EMA50 for short
        trend_long = close[i] > ema_50_1d_aligned[i]
        trend_short = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 with volume confirmation and uptrend
            if close[i] > r3 and volume_confirm and trend_long:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 with volume confirmation and downtrend
            elif close[i] < s3 and volume_confirm and trend_short:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when price returns to Camarilla H3 level (profit target) or breaks below L3 (stop)
            # Calculate H3 and L3 for exit
            h3 = prev_close + (range_val * 1.1 / 2)
            l3 = prev_close - (range_val * 1.1 / 2)
            
            if close[i] <= h3 or close[i] < l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position - exit conditions
            # Exit when price returns to Camarilla L3 level (profit target) or breaks above H3 (stop)
            h3 = prev_close + (range_val * 1.1 / 2)
            l3 = prev_close - (range_val * 1.1 / 2)
            
            if close[i] >= l3 or close[i] > h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals