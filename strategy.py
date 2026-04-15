#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w trend filter and volume spike
# Uses weekly trend for bias, daily pivot for structure, and volume confirmation
# Designed to work in both bull (breakouts with trend) and bear (faded breaks against trend)
# Target: 12-37 trades/year (50-150 total over 4 years) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly HTF data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily HTF data once for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter (needs extra delay for confirmation)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w, additional_delay_bars=1)
    
    # Daily ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Daily Camarilla pivot levels (using prior day's OHLC)
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    camarilla_pivot = (prior_high + prior_low + prior_close) / 3.0
    camarilla_r3 = camarilla_pivot + 1.1 * (prior_high - prior_low)
    camarilla_s3 = camarilla_pivot - 1.1 * (prior_high - prior_low)
    
    # Align Camarilla levels to 12h
    camarilla_pivot_12h = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(camarilla_pivot_12h[i]) or np.isnan(camarilla_r3_12h[i]) or 
            np.isnan(camarilla_s3_12h[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when daily ATR is elevated (> 0.5% of price)
        vol_regime = atr_14_1d_aligned[i] > 0.005 * close[i]
        
        # Trend filter: weekly EMA50 slope (rising/falling)
        if i >= 101:
            ema_now = ema_50_1w_aligned[i]
            ema_prev = ema_50_1w_aligned[i-1]
            trend_up = ema_now > ema_prev
            trend_down = ema_now < ema_prev
        else:
            trend_up = trend_down = False
        
        # Long conditions:
        # 1. Weekly trend up (EMA50 rising)
        # 2. Price breaks above Camarilla R3 with volume
        # 3. Volume confirmation: volume > 2.0x average
        # 4. Daily volatility regime filter
        if (trend_up and
            close[i] > camarilla_r3_12h[i] and
            volume_ratio[i] > 2.0 and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Weekly trend down (EMA50 falling)
        # 2. Price breaks below Camarilla S3 with volume
        # 3. Volume confirmation: volume > 2.0x average
        # 4. Daily volatility regime filter
        elif (trend_down and
              close[i] < camarilla_s3_12h[i] and
              volume_ratio[i] > 2.0 and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Vol_Regime_Camarilla_Pivot_R3S3_Breakout_v5"
timeframe = "12h"
leverage = 1.0