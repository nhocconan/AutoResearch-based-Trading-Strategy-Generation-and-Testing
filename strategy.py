#!/usr/bin/env python3
# 1h_VWAP_RSI_Confluence_Strategy
# Hypothesis: Combines VWAP mean reversion with RSI momentum on 1h timeframe, using 4h trend filter and volume confirmation. VWAP acts as dynamic support/resistance, RSI filters momentum strength, and 4h trend ensures alignment with higher timeframe momentum. Designed for 15-30 trades/year to avoid fee drag, effective in both trending and ranging markets.

name = "1h_VWAP_RSI_Confluence_Strategy"
timeframe = "1h"
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
    
    # Calculate VWAP (typical price * volume) cumulative
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator > 0, vwap_numerator / vwap_denominator, 0.0)
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing
    alpha = 1.0 / 14.0
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, len(gain)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # 4h trend filter: EMA20 on 4h close
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_4h_up = close_4h > ema20_4h
    trend_4h_down = close_4h < ema20_4h
    
    # Align 4h trend to 1h
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up.astype(float))
    trend_4h_down_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_down.astype(float))
    
    # Volume confirmation: 20-period average
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap[i]) or np.isnan(rsi[i]) or
            np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_4h_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: require above average volume
        volume_confirm = volume[i] > vol_ma[i] * 1.2
        
        if position == 0:
            # Enter long: price below VWAP (mean reversion) + RSI oversold + 4h uptrend
            if (close[i] < vwap[i] * 0.998 and  # 0.2% below VWAP
                rsi[i] < 35 and
                trend_4h_up_aligned[i] > 0.5 and
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Enter short: price above VWAP + RSI overbought + 4h downtrend
            elif (close[i] > vwap[i] * 1.002 and  # 0.2% above VWAP
                  rsi[i] > 65 and
                  trend_4h_down_aligned[i] > 0.5 and
                  volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns to VWAP or RSI overbought
            if (close[i] > vwap[i] * 1.001 or  # 0.1% above VWAP
                rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns to VWAP or RSI oversold
            if (close[i] < vwap[i] * 0.999 or  # 0.1% below VWAP
                rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals