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
    
    # Get weekly data for higher timeframe context (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Calculate weekly EMA(34) for trend direction
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly volume moving average
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Calculate daily ATR(14) for volatility filter (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    tr1_d = high_1d - low_1d
    tr2_d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_d[0] = tr1_d[0]
    tr3_d[0] = tr1_d[0]
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    atr_14_1d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily EMA(34) for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily volume moving average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i]) or 
            np.isnan(vol_ma_1w_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above/below weekly EMA34
        price_above_weekly_ema = close[i] > ema_34_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_34_1w_aligned[i]
        
        # Daily trend filter: price above/below daily EMA34
        price_above_daily_ema = close[i] > ema_34_1d_aligned[i]
        price_below_daily_ema = close[i] < ema_34_1d_aligned[i]
        
        # Volatility filter: avoid high volatility periods (weekly ATR below median)
        atr_median_w = np.nanmedian(atr_14_1w_aligned[:i+1]) if i >= 50 else atr_14_1w_aligned[i]
        low_volatility_w = atr_14_1w_aligned[i] < atr_median_w
        
        # Daily volatility filter
        atr_median_d = np.nanmedian(atr_14_1d_aligned[:i+1]) if i >= 50 else atr_14_1d_aligned[i]
        low_volatility_d = atr_14_1d_aligned[i] < atr_median_d
        
        # Volume filter: current volume above weekly average
        volume_filter_w = volume[i] > vol_ma_1w_aligned[i]
        
        # Daily volume filter
        volume_filter_d = volume[i] > vol_ma_1d_aligned[i]
        
        # Long conditions: price above both EMAs + low volatility + volume
        long_condition = (price_above_weekly_ema and 
                         price_above_daily_ema and 
                         low_volatility_w and 
                         low_volatility_d and 
                         volume_filter_w and
                         volume_filter_d)
        
        # Short conditions: price below both EMAs + low volatility + volume
        short_condition = (price_below_weekly_ema and 
                          price_below_daily_ema and 
                          low_volatility_w and 
                          low_volatility_d and 
                          volume_filter_w and
                          volume_filter_d)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or volatility spike
        elif position == 1 and (not price_above_weekly_ema or not price_above_daily_ema or not low_volatility_w or not low_volatility_d):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not price_below_weekly_ema or not price_below_daily_ema or not low_volatility_w or not low_volatility_d):
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

name = "1d_DualEMA34_VolumeFilter_1w1dTrend_Session"
timeframe = "1d"
leverage = 1.0