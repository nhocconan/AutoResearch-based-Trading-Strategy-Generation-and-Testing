#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsVixFix_MeanReversion_1dTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Vix Fix and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 22:
        return np.zeros(n)
    
    # Williams Vix Fix on daily (mean reversion signal)
    # WVF = ((Highest Close in period - Low) / Highest Close in period) * 100
    lookback = 22
    highest_close = pd.Series(df_1d['close']).rolling(window=lookback, min_periods=lookback).max().values
    wvf = ((highest_close - df_1d['low'].values) / highest_close) * 100
    wvf_mean = pd.Series(wvf).rolling(window=lookback, min_periods=lookback).mean().values
    wvf_std = pd.Series(wvf).rolling(window=lookback, min_periods=lookback).std().values
    wvf_zscore = (wvf - wvf_mean) / wvf_std
    
    # Align WVF z-score to 6h
    wvf_zscore_6h = align_htf_to_ltf(prices, df_1d, wvf_zscore)
    
    # 1d EMA50 for trend filter (avoid counter-trend in strong trends)
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h RSI for entry timing (avoid extremes)
    rsi_period = 14
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(wvf_zscore_6h[i]) or np.isnan(ema50_6h[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Mean reversion conditions
        wvf_high = wvf_zscore_6h[i] > 2.0  # High fear = potential bounce
        wvf_low = wvf_zscore_6h[i] < -2.0  # Low fear = potential fade
        rsi_not_extreme = (rsi[i] > 20) & (rsi[i] < 80)  # Avoid RSI extremes
        
        if position == 0:
            # Long: High fear (WVF spike) + above EMA50 (bullish bias) + RSI not oversold
            if wvf_high and close[i] > ema50_6h[i] and rsi_not_extreme:
                signals[i] = 0.25
                position = 1
            # Short: Low fear (WVF drop) + below EMA50 (bearish bias) + RSI not overbought
            elif wvf_low and close[i] < ema50_6h[i] and rsi_not_extreme:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Fear subsides OR RSI overbought OR trend turns
            if (wvf_zscore_6h[i] < 0) or (rsi[i] > 70) or (close[i] < ema50_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Fear subsides OR RSI oversold OR trend turns
            if (wvf_zscore_6h[i] > 0) or (rsi[i] < 30) or (close[i] > ema50_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals