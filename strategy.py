#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot R3/S3 breakout with 1w trend filter and volume confirmation
# Uses higher timeframe (1w) for trend bias to avoid counter-trend trades in choppy markets
# Volume confirmation ensures breakouts have participation
# Discrete position sizing (0.25) to limit fee churn
# Designed to work in both bull (trend-following breaks) and bear (mean reversion at extremes) markets
# Target: 15-25 trades/year to stay within fee drag limits

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA(21) for trend filter
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate daily ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily Camarilla pivot levels (using prior day's OHLC)
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    camarilla_pivot = (prior_high + prior_low + prior_close) / 3.0
    camarilla_r3 = camarilla_pivot + 1.1 * (prior_high - prior_low)
    camarilla_s3 = camarilla_pivot - 1.1 * (prior_high - prior_low)
    camarilla_r4 = camarilla_pivot + 1.5 * (prior_high - prior_low)
    camarilla_s4 = camarilla_pivot - 1.5 * (prior_high - prior_low)
    
    # Align Camarilla levels to 1d
    camarilla_pivot_1d = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_1d = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_1d = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_1d = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_1d = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(camarilla_pivot_1d[i]) or np.isnan(camarilla_r3_1d[i]) or 
            np.isnan(camarilla_s3_1d[i]) or np.isnan(camarilla_r4_1d[i]) or 
            np.isnan(camarilla_s4_1d[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when daily ATR is elevated (> 0.6% of price)
        # This avoids low-volatility chop and focuses on momentum/trend days
        vol_regime = atr_14_1d_aligned[i] > 0.006 * close[i]
        
        # Long conditions:
        # 1. Price above 1w EMA21 (bullish trend bias)
        # 2. Price breaks above Camarilla R3 with volume (bullish continuation)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Daily volatility regime filter (avoid chop)
        if (close[i] > ema_21_1w_aligned[i] and
            close[i] > camarilla_r3_1d[i] and
            volume_ratio[i] > 1.5 and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below 1w EMA21 (bearish trend bias)
        # 2. Price breaks below Camarilla S3 with volume (bearish continuation)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Daily volatility regime filter
        elif (close[i] < ema_21_1w_aligned[i] and
              close[i] < camarilla_s3_1d[i] and
              volume_ratio[i] > 1.5 and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Camarilla_R3S3_1wEMA21_Volume_Regime_v1"
timeframe = "1d"
leverage = 1.0