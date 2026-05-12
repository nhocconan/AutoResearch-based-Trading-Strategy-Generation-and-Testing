#!/usr/bin/env python3
# 4h_1D_Keltner_RSI_Fade_1dTrend_VolumeFilter
# Hypothesis: Fade extreme moves from Keltner Bands using RSI(2) and daily trend filter.
# Long: Price touches lower Keltner band + RSI(2) < 10 + price above daily EMA34 (uptrend).
# Short: Price touches upper Keltner band + RSI(2) > 90 + price below daily EMA34 (downtrend).
# Exit when price crosses daily EMA34. Uses volume spike to confirm exhaustion.
# Designed for mean reversion in ranging markets and pullback entries in trends.
# Targets 20-50 trades per year with strict confluence.

name = "4h_1D_Keltner_RSI_Fade_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Keltner Bands: EMA(20) ± ATR(10)*2
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_10 = pd.Series(high - low).rolling(window=10, min_periods=10).mean().values
    upper_keltner = ema_20 + 2 * atr_10
    lower_keltner = ema_20 - 2 * atr_10
    
    # RSI(2) for extreme readings
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss
    rsi_2 = 100 - (100 / (1 + rs))
    rsi_2 = rsi_2.fillna(50).values  # neutral when undefined
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price at lower Keltner + RSI(2) oversold + uptrend + volume spike
            if (close[i] <= lower_keltner[i] and 
                rsi_2[i] < 10 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at upper Keltner + RSI(2) overbought + downtrend + volume spike
            elif (close[i] >= upper_keltner[i] and 
                  rsi_2[i] > 90 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above daily EMA34 (trend continuation)
            if close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below daily EMA34 (trend continuation)
            if close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals