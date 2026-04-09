#!/usr/bin/env python3
# 1d_ema_bull_bear_flip_v1
# Hypothesis: Uses 1-day EMA trend with 21/55 crossover for trend detection, combined with
# 1-week RSI for overbought/oversold conditions and volume confirmation. In bull markets
# (price > EMA21), looks for RSI pullbacks to go long. In bear markets (price < EMA55),
# looks for RSI bounces to go short. Designed to work in both bull and bear regimes by
# adapting strategy based on longer-term trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ema_bull_bear_flip_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1. 1-day EMA21 and EMA55 for trend determination
    close_series = pd.Series(close)
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).values
    ema55 = close_series.ewm(span=55, adjust=False, min_periods=55).values
    
    # 2. 1-week RSI for overbought/oversold conditions
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate RSI on weekly data
    rsi_period = 14
    delta = pd.Series(df_1w['close']).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period, min_periods=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period, min_periods=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align weekly RSI to daily timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi_values)
    
    # 3. Volume confirmation (20-day average)
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(55, n):  # Start after EMA55 warmup
        # Skip if any required data is invalid
        if (np.isnan(ema21[i]) or np.isnan(ema55[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: price closes below EMA21 OR RSI becomes overbought (>70)
            if close[i] < ema21[i] or rsi_aligned[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above EMA55 OR RSI becomes oversold (<30)
            if close[i] > ema55[i] or rsi_aligned[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Determine market regime based on EMA relationship
            bull_market = ema21[i] > ema55[i]
            bear_market = ema21[i] < ema55[i]
            
            # Enter long in bull market: price pulls back to EMA21 with RSI oversold
            if bull_market and vol_ok:
                if close[i] <= ema21[i] * 1.01 and rsi_aligned[i] < 35:
                    position = 1
                    signals[i] = 0.25
            # Enter short in bear market: price bounces to EMA55 with RSI overbought
            elif bear_market and vol_ok:
                if close[i] >= ema55[i] * 0.99 and rsi_aligned[i] > 65:
                    position = -1
                    signals[i] = -0.25
    
    return signals