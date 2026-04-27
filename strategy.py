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
    
    # Get 1d data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ADX for trend strength (no look-ahead)
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                              np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.nan], tr])
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx[np.isnan(dx)] = np.nan
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1d RSI for momentum
    delta_1d = pd.Series(close_1d).diff()
    gain_1d = delta_1d.where(delta_1d > 0, 0)
    loss_1d = -delta_1d.where(delta_1d < 0, 0)
    avg_gain_1d = gain_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1d = loss_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1d = avg_gain_1d / avg_loss_1d
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d ATR for volatility filter
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                 np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d volume moving average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: strong trend (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        # Momentum filter: RSI in neutral zone (avoid extremes)
        rsi_neutral = (rsi_1d_aligned[i] > 30) & (rsi_1d_aligned[i] < 70)
        
        # Volatility filter: reasonable volatility
        vol_filter = atr_1d_aligned[i] > 0
        
        # Volume filter: current volume above average
        volume_filter = volume[i] > vol_ma_1d_aligned[i]
        
        # Long conditions: strong trend + RSI neutral + volume + volatility
        long_condition = strong_trend and rsi_neutral and volume_filter and vol_filter
        
        # Short conditions: strong trend + RSI neutral + volume + volatility
        short_condition = strong_trend and rsi_neutral and volume_filter and vol_filter
        
        # Determine trend direction using price vs 20-period SMA on 1d
        sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
        sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d)
        
        if not np.isnan(sma_20_1d_aligned[i]):
            price_above_sma = close[i] > sma_20_1d_aligned[i]
            price_below_sma = close[i] < sma_20_1d_aligned[i]
        else:
            price_above_sma = False
            price_below_sma = False
        
        # Long: uptrend (price above SMA20)
        if long_condition and price_above_sma and position <= 0:
            signals[i] = 0.25
            position = 1
        # Short: downtrend (price below SMA20)
        elif short_condition and price_below_sma and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit: trend weakening (ADX < 20) or RSI extreme
        elif position == 1 and (adx_aligned[i] < 20 or rsi_1d_aligned[i] > 70):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (adx_aligned[i] < 20 or rsi_1d_aligned[i] < 30):
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

name = "12h_ADX25_RSI_VolumeFilter_1dTrend"
timeframe = "12h"
leverage = 1.0