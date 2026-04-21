#!/usr/bin/env python3
"""
1h_Volume_Weighted_RSI_v1
Hypothesis: On 1h timeframe, use 4h EMA for trend direction and 1d RSI(14) for mean-reversion extremes.
Enter long when price > 4h EMA (uptrend) AND 1d RSI < 30 (oversold) with volume confirmation.
Enter short when price < 4h EMA (downtrend) AND 1d RSI > 70 (overbought) with volume confirmation.
Use 1h only for precise entry timing via RSI(2) pullback to reduce false signals.
Target 15-30 trades/year via tight confluence filters. Works in bull (trend follow) and bear (mean revert) regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 21 or len(df_1d) < 14:
        return np.zeros(n)
    
    # === 4h EMA21 for trend filter ===
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # === 1d RSI14 for mean reversion ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # === 1h RSI2 for entry timing ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(span=2, adjust=False, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_2 = 100 - (100 / (1 + rs))
    
    # === Volume confirmation (1h) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_21_4h_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(rsi_2[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_21 = ema_21_4h_aligned[i]
        rsi_1d = rsi_14_1d_aligned[i]
        rsi_2_val = rsi_2[i]
        vol_spike = vol_ratio[i] > 1.5  # 50% above average volume
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: Uptrend (price > 4h EMA21) + 1d RSI oversold (<30) + 1h RSI pullback (<20) + volume spike + session
            if price_close > ema_21 and rsi_1d < 30 and rsi_2_val < 20 and vol_spike and in_session:
                signals[i] = 0.20
                position = 1
            # Short: Downtrend (price < 4h EMA21) + 1d RSI overbought (>70) + 1h RSI pullback (>80) + volume spike + session
            elif price_close < ema_21 and rsi_1d > 70 and rsi_2_val > 80 and vol_spike and in_session:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: 1h RSI mean reversion or trend change
            if position == 1:
                if rsi_2_val > 80 or price_close < ema_21:  # overbought or trend break
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if rsi_2_val < 20 or price_close > ema_21:  # oversold or trend break
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Volume_Weighted_RSI_v1"
timeframe = "1h"
leverage = 1.0