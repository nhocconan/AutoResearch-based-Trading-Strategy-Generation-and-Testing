#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and ADX trend filter
# Uses discrete position sizing (0.25) to minimize fee churn
# Volume spike confirms institutional interest, ADX>25 ensures trending environment
# Works in bull/bear by trading breakouts in direction of 1d trend (via ADX)
# Target: 20-40 trades/year to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend strength filter
    # +DM, -DM, TR
    up_move = df_1d['high'].diff()
    down_move = df_1d['low'].diff().multiply(-1)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values with Welles Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    tr_sum = np.zeros_like(tr)
    plus_dm_sum = np.zeros_like(plus_dm)
    minus_dm_sum = np.zeros_like(minus_dm)
    
    # Initial values
    tr_sum[period] = tr.iloc[:period+1].sum()
    plus_dm_sum[period] = plus_dm.iloc[:period+1].sum()
    minus_dm_sum[period] = minus_dm.iloc[:period+1].sum()
    
    # Wilder's smoothing
    for i in range(period + 1, len(tr)):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / period) + tr.iloc[i]
        plus_dm_sum[i] = plus_dm_sum[i-1] - (plus_dm_sum[i-1] / period) + plus_dm.iloc[i]
        minus_dm_sum[i] = minus_dm_sum[i-1] - (minus_dm_sum[i-1] / period) + minus_dm.iloc[i]
    
    # Avoid division by zero
    tr_sum[tr_sum == 0] = 1e-10
    
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    
    # ADX: smoothed DX
    adx = np.zeros_like(dx)
    adx[2*period] = dx.iloc[period:2*period+1].mean()
    for i in range(2*period + 1, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx.iloc[i]) / period
    
    adx_values = adx
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Calculate 1d ATR(14) for volatility filter
    atr_14_1d = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily Camarilla pivot levels (using prior day's OHLC)
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    camarilla_pivot = (prior_high + prior_low + prior_close) / 3.0
    camarilla_r3 = camarilla_pivot + 1.1 * (prior_high - prior_low)
    camarilla_s3 = camarilla_pivot - 1.1 * (prior_high - prior_low)
    
    # Align Camarilla levels to 4h
    camarilla_pivot_4h = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 4h ATR(14) for volatility entry filter
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_4h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(camarilla_pivot_4h[i]) or np.isnan(camarilla_r3_4h[i]) or 
            np.isnan(camarilla_s3_4h[i]) or np.isnan(atr_14_4h[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when ADX > 25 (strong trend)
        trend_filter = adx_aligned[i] > 25
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_14_1d_aligned[i] > 0.005 * close[i]
        
        # Long conditions:
        # 1. Price breaks above Camarilla R3 with volume
        # 2. Volume confirmation: volume > 1.5x average
        # 3. ATR > 0.3% of price (ensure sufficient volatility for move)
        # 4. Trend and volatility filters
        if (close[i] > camarilla_r3_4h[i] and
            volume_ratio[i] > 1.5 and
            atr_14_4h[i] > 0.003 * close[i] and
            trend_filter and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below Camarilla S3 with volume
        # 2. Volume confirmation: volume > 1.5x average
        # 3. ATR > 0.3% of price
        # 4. Trend and volatility filters
        elif (close[i] < camarilla_s3_4h[i] and
              volume_ratio[i] > 1.5 and
              atr_14_4h[i] > 0.003 * close[i] and
              trend_filter and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_Volume_ADX_v1"
timeframe = "4h"
leverage = 1.0