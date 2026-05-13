#!/usr/bin/env python3
# Hypothesis: 1h mean reversion with 4h trend filter and volume confirmation. 
# Long when price pulls back to 4h VWAP with RSI(14) < 30 and volume > 1.5x average in uptrend (4h close > 4h EMA20).
# Short when price rallies to 4h VWAP with RSI(14) > 70 and volume > 1.5x average in downtrend (4h close < 4h EMA20).
# Uses discrete sizing 0.20. Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
# 4h EMA20 filters for intermediate trend, 4h VWAP provides dynamic support/resistance, RSI identifies overextended moves.
# Volume confirmation ensures participation. Works in bull markets via pullback longs and in bear markets via rally shorts.

name = "1h_VWAP_RSI_MeanReversion_4hEMA20_VolumeConfirm_v1"
timeframe = "1h"
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
    
    # Calculate 1h VWAP (typical price * volume) cumulative
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 4h data for EMA20 trend filter and VWAP
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h EMA20
    ema_20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Calculate 4h VWAP
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    vwap_num_4h = np.cumsum(typical_price_4h * volume_4h)
    vwap_den_4h = np.cumsum(volume_4h)
    vwap_4h = vwap_num_4h / vwap_den_4h
    
    # Align 4h indicators to 1h timeframe (wait for 4h bar to close)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
    
    # Calculate average volume for confirmation (24-period)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(vwap[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(vwap_4h_aligned[i]) or 
            np.isnan(close_4h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price near 1h VWAP, RSI oversold, 4h uptrend, volume spike
            if (close[i] <= vwap[i] * 1.005 and  # Within 0.5% above VWAP
                rsi[i] < 30 and 
                close_4h_aligned[i] > ema_20_4h_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price near 1h VWAP, RSI overbought, 4h downtrend, volume spike
            elif (close[i] >= vwap[i] * 0.995 and  # Within 0.5% below VWAP
                  rsi[i] > 70 and 
                  close_4h_aligned[i] < ema_20_4h_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches VWAP or RSI overbought
            if close[i] >= vwap[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price reaches VWAP or RSI oversold
            if close[i] <= vwap[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals