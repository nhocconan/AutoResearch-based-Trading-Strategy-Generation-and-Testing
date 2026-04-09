#!/usr/bin/env python3
# 1d_keltner_channel_v1
# Hypothesis: 1d strategy using Keltner Channel (20, ATR=2.0) breakouts with volume confirmation.
# Long when price closes above upper band + volume > 1.5x 20-day average.
# Short when price closes below lower band + volume > 1.5x 20-day average.
# Uses 1w HTF EMA(50) as trend filter: only long when price > weekly EMA50, short when price < weekly EMA50.
# Discrete sizing (0.0, ±0.25) to minimize fee churn. Target: 15-25 trades/year.
# Works in bull/bear: breaks capture momentum, weekly EMA filter avoids counter-trend trades.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_keltner_channel_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Keltner Channel (20, ATR=2.0)
    atr_period = 20
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_band = ema_20 + 2.0 * atr
    lower_band = ema_20 - 2.0 * atr
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_20[i]) or np.isnan(atr[i]) or np.isnan(upper_band[i]) or
            np.isnan(lower_band[i]) or np.isnan(volume_ma[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below middle line (EMA20) OR weekly trend turns bearish
            if close[i] < ema_20[i] or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above middle line (EMA20) OR weekly trend turns bullish
            if close[i] > ema_20[i] or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: price closes above upper band AND price > weekly EMA50 (bullish trend)
                if close[i] > upper_band[i] and close[i] > ema_50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price closes below lower band AND price < weekly EMA50 (bearish trend)
                elif close[i] < lower_band[i] and close[i] < ema_50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals