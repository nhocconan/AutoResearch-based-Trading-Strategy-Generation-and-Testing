#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: Daily Bollinger Band squeeze breakout with weekly trend filter and volume confirmation
    # Bollinger squeeze identifies low volatility periods preceding breakouts
    # Weekly EMA50 determines primary trend direction for bias
    # Volume spike confirms institutional participation in breakout direction
    # Works in both bull and bear markets by trading volatility breakouts in trend direction
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands on daily data
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bollinger Bands (20, 2)
    sma20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    bb_width = (upper_bb - lower_bb) / sma20  # Normalized width
    
    # Bollinger Squeeze: BB width below 20-period average indicates low volatility
    bb_width_ma20 = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma20  # Squeeze condition
    
    # Align squeeze signal to daily timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze)
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily close for trend comparison
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume spike filter (20-period on daily volume)
    volume_1d = df_1d['volume'].values
    vol_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > 2.0 * vol_ma20  # Require 2x volume for confirmation
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(squeeze_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(close_1d_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bollinger breakout above upper band with volume spike and price above weekly EMA50 (uptrend)
            if (close[i] > upper_bb[i] and 
                squeeze_aligned[i] and 
                vol_spike_aligned[i] and 
                close_1d_aligned[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bollinger breakout below lower band with volume spike and price below weekly EMA50 (downtrend)
            elif (close[i] < lower_bb[i] and 
                  squeeze_aligned[i] and 
                  vol_spike_aligned[i] and 
                  close_1d_aligned[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to middle Bollinger Band (mean reversion after breakout)
            if position == 1:
                if close[i] < sma20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > sma20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Bollinger_Squeeze_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0