# 1d_1w_RSI_Momentum_With_Trend_Filter
# Uses weekly RSI for momentum bias and daily RSI for entry timing
# Weekly RSI > 60 = bullish bias (longs), < 40 = bearish bias (shorts)
# Daily RSI < 30 for long entry in bullish bias, > 70 for short entry in bearish bias
# Volume confirmation and volatility filter to reduce false signals
# Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag
# Works in both bull and bear markets via weekly trend filter

#!/usr/bin/env python3
name = "1d_1w_RSI_Momentum_With_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly RSI for momentum bias (14-period)
    weekly_close = pd.Series(df_1w['close'])
    delta = weekly_close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_weekly = rsi_14.values
    
    # Align weekly RSI to daily timeframe
    rsi_weekly_aligned = align_htf_to_ltf(prices, df_1w, rsi_weekly)
    
    # Daily RSI for entry timing (14-period)
    daily_close_series = pd.Series(close)
    delta_daily = daily_close_series.diff()
    gain_daily = (delta_daily.where(delta_daily > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss_daily = (-delta_daily.where(delta_daily < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs_daily = gain_daily / loss_daily
    rsi_daily = 100 - (100 / (1 + rs_daily))
    
    # Volatility filter: ATR(14) normalized by price
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    volatility_normalized = atr / close
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Wait for volatility and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_weekly_aligned[i]) or np.isnan(rsi_daily[i]) or 
            np.isnan(volatility_normalized[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade when volatility is reasonable (not too high, not too low)
        vol_filter = (volatility_normalized[i] > 0.01) & (volatility_normalized[i] < 0.08)
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long signal: bullish weekly bias + oversold daily RSI
            bullish_bias = rsi_weekly_aligned[i] > 60
            oversold = rsi_daily[i] < 30
            
            if bullish_bias and oversold and vol_filter and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short signal: bearish weekly bias + overbought daily RSI
            elif rsi_weekly_aligned[i] < 40 and rsi_daily[i] > 70 and vol_filter and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI becomes overbought or weekly bias turns bearish
            if rsi_daily[i] > 70 or rsi_weekly_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI becomes oversold or weekly bias turns bullish
            if rsi_daily[i] < 30 or rsi_weekly_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals