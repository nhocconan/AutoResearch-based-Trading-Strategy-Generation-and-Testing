#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d Williams %R extremes (overbought/oversold) 
# combined with volume confirmation and trend filter from 1d EMA50.
# Williams %R identifies reversal points in ranging markets, while EMA50 filter
# avoids counter-trend trades in strong trends. Volume confirms institutional interest.
# Designed for both ranging (mean reversion) and trending (breakout continuation) markets.

name = "6h_1d_WilliamsR_Volume_EMA50"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.full_like(close_1d, np.nan)
    for i in range(13, len(close_1d)):
        highest_high = np.max(high_1d[i-13:i+1])
        lowest_low = np.min(low_1d[i-13:i+1])
        if highest_high > lowest_low:
            williams_r[i] = ((highest_high - close_1d[i]) / (highest_high - lowest_low)) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Calculate EMA50 on daily close for trend filter
    close_series = pd.Series(close_1d)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily average volume (20-period) for volume spike detection
    vol_1d = df_1d['volume'].values
    vol_series = pd.Series(vol_1d)
    vol_avg = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all daily indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    
    # Calculate 6-day average true range for volatility filter (optional)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=6, min_periods=6).mean().values  # 6-period ATR for 6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        current_atr = atr[i]
        
        # Get aligned daily values
        wr = williams_r_aligned[i]
        ema = ema50_aligned[i]
        vol_avg_val = vol_avg_aligned[i]
        
        # Skip if any data is not available
        if np.isnan(wr) or np.isnan(ema) or np.isnan(vol_avg_val):
            continue
        
        # Volume spike: current volume > 1.8x daily average volume
        vol_spike = current_volume > 1.8 * vol_avg_val
        
        # Williams %R levels: oversold < -80, overbought > -20
        wr_oversold = wr < -80
        wr_overbought = wr > -20
        
        # Trend filter: price above EMA50 = uptrend, below = downtrend
        price_above_ema = current_close > ema
        price_below_ema = current_close < ema
        
        if position == 0:
            # Long setup: Williams %R oversold + volume spike + price above EMA (buy dip in uptrend)
            if wr_oversold and vol_spike and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short setup: Williams %R overbought + volume spike + price below EMA (sell rally in downtrend)
            elif wr_overbought and vol_spike and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R overbought (overbought condition) or trend change
            if wr_overbought or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R oversold (oversold condition) or trend change
            if wr_oversold or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals