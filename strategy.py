#!/usr/bin/env python3
"""
Hypothesis: 1h RSI divergence with 4h trend filter and volume confirmation.
In ranging markets, RSI divergence at extremes signals reversals. Trend filter ensures
we only take reversals in the direction of higher timeframe momentum to avoid
counter-trend trades. Volume confirmation filters weak signals. Designed for 15-30
trades/year to minimize fee drag, works in bull/bear via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA34 for trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Load 1d data ONCE before loop for RSI(14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # RSI(14) calculation
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation: volume / 20-period average volume (1d)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_34_4h_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        vol_threshold = 1.3  # Volume must be 1.3x average
        
        if position == 0:
            # Enter long: RSI oversold (<30) with bullish divergence, volume spike, uptrend
            # Bullish divergence: price making lower low, RSI making higher low
            if i >= 2:
                price_lower_low = prices['close'].iloc[i] < prices['close'].iloc[i-2]
                rsi_higher_low = rsi_val > rsi_aligned[i-2]
                if (rsi_val < 30 and 
                    price_lower_low and 
                    rsi_higher_low and
                    vol_ratio > vol_threshold and 
                    price_close > ema_trend):
                    signals[i] = 0.20
                    position = 1
            # Enter short: RSI overbought (>70) with bearish divergence, volume spike, downtrend
            elif i >= 2:
                price_higher_high = prices['close'].iloc[i] > prices['close'].iloc[i-2]
                rsi_lower_high = rsi_val < rsi_aligned[i-2]
                if (rsi_val > 70 and 
                    price_higher_high and 
                    rsi_lower_high and
                    vol_ratio > vol_threshold and 
                    price_close < ema_trend):
                    signals[i] = -0.20
                    position = -1
        
        elif position != 0:
            # Exit: RSI returns to neutral zone (40-60) or trend reversal
            if position == 1 and (rsi_val > 50 or price_close < ema_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (rsi_val < 50 or price_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RSIDivergence_4hEMA34_Trend_Volume"
timeframe = "1h"
leverage = 1.0