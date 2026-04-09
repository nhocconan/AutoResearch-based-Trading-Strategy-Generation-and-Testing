#!/usr/bin/env python3
# 1h_ema_rsi_trend_4h1d_v1
# Hypothesis: 1h EMA trend aligned with 4h EMA and 1d regime filter (chop). Uses volume confirmation for entries.
# In bull/bear markets, price respects EMA alignment; in ranging markets (2025+), chop filter avoids false signals.
# Volume confirmation ensures momentum validity. Discrete sizing (0.0, ±0.20) minimizes fee churn.
# Target: 60-150 total trades over 4 years by requiring EMA alignment + volume + chop regime filter.
# Primary timeframe: 1h, HTF: 4h for trend, 1d for chop regime.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_ema_rsi_trend_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h HTF data for EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # 4h EMA(21) for trend
    close_4h = pd.Series(df_4h['close'].values)
    ema_4h = close_4h.ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d HTF data for chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d Chopiness Index (14-period)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    atr_1d = high_1d - low_1d
    high_14 = high_1d.rolling(window=14, min_periods=14).max().values
    low_14 = low_1d.rolling(window=14, min_periods=14).min().values
    atr_sum = atr_1d.rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero and log(0)
    chop_denom = np.log10(atr_sum) * np.log10(14)
    chop_denom = np.where(chop_denom <= 0, 1e-10, chop_denom)
    chop = 100 * np.log10((high_14 - low_14) / chop_denom) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 1h RSI(14) for momentum
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 100, rs)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # 1h volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(rsi_values[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        in_session = (8 <= hours[i] <= 20)
        volume_confirmed = volume[i] > 1.2 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: EMA trend turns bearish OR RSI overbought OR volume dries up OR out of session
            if (close[i] < ema_4h_aligned[i] or 
                rsi_values[i] > 70 or 
                not volume_confirmed or 
                not in_session):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: EMA trend turns bullish OR RSI oversold OR volume dries up OR out of session
            if (close[i] > ema_4h_aligned[i] or 
                rsi_values[i] < 30 or 
                not volume_confirmed or 
                not in_session):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            if in_session and volume_confirmed:
                # Long entry: price above 4h EMA AND RSI bullish AND chop > 50 (ranging/transition)
                if (close[i] > ema_4h_aligned[i] and 
                    rsi_values[i] > 50 and 
                    rsi_values[i] < 70 and 
                    chop_aligned[i] > 50):
                    position = 1
                    signals[i] = 0.20
                # Short entry: price below 4h EMA AND RSI bearish AND chop > 50
                elif (close[i] < ema_4h_aligned[i] and 
                      rsi_values[i] < 50 and 
                      rsi_values[i] > 30 and 
                      chop_aligned[i] > 50):
                    position = -1
                    signals[i] = -0.20
    
    return signals