#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly RSI(14) for trend strength
    delta = pd.Series(close_1w).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi_14_1w = (100 - (100 / (1 + rs))).values
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    
    # Calculate daily ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily Bollinger Bands (20, 2) for mean reversion
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20_1d + (2 * std_20_1d)
    lower_bb = sma_20_1d - (2 * std_20_1d)
    sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Calculate daily volume moving average for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_14_1w_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(sma_20_1d_aligned[i]) or
            np.isnan(upper_bb_aligned[i]) or
            np.isnan(lower_bb_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: weekly RSI > 50 for bullish, < 50 for bearish
        bullish_trend = rsi_14_1w_aligned[i] > 50
        bearish_trend = rsi_14_1w_aligned[i] < 50
        
        # Volatility filter: current ATR > average ATR
        vol_filter = atr_14_1d_aligned[i] > 0 and close[i] > atr_14_1d_aligned[i] * 0.1
        
        # Mean reversion signals: price touches Bollinger Bands
        touch_upper = close[i] >= upper_bb_aligned[i]
        touch_lower = close[i] <= lower_bb_aligned[i]
        
        # Volume filter: current volume above average
        volume_filter = vol_ma_20_1d_aligned[i] > 0 and volume[i] > vol_ma_20_1d_aligned[i] * 1.5
        
        # Long conditions: bullish trend + volatility + volume + touch lower BB (mean reversion long)
        long_condition = (bullish_trend and 
                         vol_filter and 
                         volume_filter and 
                         touch_lower)
        
        # Short conditions: bearish trend + volatility + volume + touch upper BB (mean reversion short)
        short_condition = (bearish_trend and 
                          vol_filter and 
                          volume_filter and 
                          touch_upper)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to middle (SMA) or trend reversal
        elif position == 1 and (close[i] >= sma_20_1d_aligned[i] or not bullish_trend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] <= sma_20_1d_aligned[i] or not bearish_trend):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WRSI14_BBands_MeanReversion_VolumeFilter"
timeframe = "1d"
leverage = 1.0