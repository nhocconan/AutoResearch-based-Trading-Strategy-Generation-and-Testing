#!/usr/bin/env python3
# 12h_RSI_Trend_Filter
# Hypothesis: Uses 12h RSI with 1d trend filter for mean reversion in ranging markets.
# Long when: 1) 1d trend is bullish (price > EMA50), 2) 12h RSI < 30 (oversold), 3) Volume > 1.2x 20-period average.
# Short when: 1) 1d trend is bearish (price < EMA50), 2) 12h RSI > 70 (overbought), 3) Volume > 1.2x 20-period average.
# Exit when RSI returns to neutral (40-60) or trend flips.
# Works in bull markets by buying dips in uptrends and in bear markets by selling rallies in downtrends.
# RSI provides mean-reversion signals, trend filter avoids counter-trend trades, volume confirms momentum.

name = "12h_RSI_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d EMA50 for trend filter ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # --- 12h RSI(14) ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    avg_gain = gain_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 50.0), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # --- 12h volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1d EMA50 to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for RSI(14) and volume MA(20)
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi[i]) or
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend from 1d EMA50
        is_uptrend = close[i] > ema50_1d_aligned[i]
        is_downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.2  # 20% above average
        
        if position == 0:
            if is_uptrend and rsi[i] < 30 and vol_spike:
                # Long: uptrend + oversold RSI + volume spike
                signals[i] = 0.25
                position = 1
            elif is_downtrend and rsi[i] > 70 and vol_spike:
                # Short: downtrend + overbought RSI + volume spike
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: RSI returns to neutral OR trend breaks down
                if rsi[i] > 40 or not is_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI returns to neutral OR trend breaks up
                if rsi[i] < 60 or not is_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals