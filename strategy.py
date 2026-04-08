#!/usr/bin/env python3
# 1d_engulfing_volume_trend_v1
# Hypothesis: Daily bullish/bearish engulfing candles with volume confirmation and weekly trend filter.
# Works in bull/bear: Engulfing captures strong momentum shifts, weekly trend ensures alignment with higher timeframe direction,
# volume filter ensures institutional participation. Low frequency (~10-20 trades/year) minimizes fee drag.

name = "1d_engulfing_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly trend filter: EMA(21) on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: volume > 1.5x 20-day average
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Engulfing candle detection
    bullish_engulf = np.zeros(n, dtype=bool)
    bearish_engulf = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        # Bullish engulfing: current green candle completely engulfs previous red candle
        if (close[i] > open_price[i] and  # current bullish
            open_price[i-1] > close[i-1] and  # previous bearish
            open_price[i] <= close[i-1] and  # current open <= previous close
            close[i] >= open_price[i-1]):  # current close >= previous open
            bullish_engulf[i] = True
        
        # Bearish engulfing: current red candle completely engulfs previous green candle
        elif (close[i] < open_price[i] and  # current bearish
              open_price[i-1] < close[i-1] and  # previous bullish
              open_price[i] >= close[i-1] and  # current open >= previous close
              close[i] <= open_price[i-1]):  # current close <= previous open
            bearish_engulf[i] = True
    
    # Start from sufficient lookback
    start_idx = max(21, vol_period) + 1
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit on bearish engulfing or trend failure
            if bearish_engulf[i] or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on bullish engulfing or trend failure
            if bullish_engulf[i] or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: bullish engulfing with uptrend and volume
            if bullish_engulf[i] and close[i] > ema_1w_aligned[i] and volume_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: bearish engulfing with downtrend and volume
            elif bearish_engulf[i] and close[i] < ema_1w_aligned[i] and volume_filter:
                position = -1
                signals[i] = -0.25
    
    return signals