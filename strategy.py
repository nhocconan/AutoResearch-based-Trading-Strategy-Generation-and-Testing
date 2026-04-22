#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and volume confirmation
# In ranging markets (common in 2025-2026 BTC/ETH), RSI extremes often revert.
# 4h EMA(50) filter ensures we only trade against the higher timeframe trend
# (mean reversion in rallies, continuation in dips) to avoid trend-following whipsaws.
# Volume confirmation (>1.3x 20-period average) filters low-conviction moves.
# Designed for 1h timeframe targeting 15-35 trades/year (~60-140 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # RSI(14) on 1h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # warmup for RSI and EMA
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) + 4h uptrend (price > EMA) + volume confirmation
            if (rsi[i] < 30 and
                close[i] > ema_50_4h_aligned[i] and
                volume[i] > 1.3 * vol_avg_20[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought (>70) + 4h downtrend (price < EMA) + volume confirmation
            elif (rsi[i] > 70 and
                  close[i] < ema_50_4h_aligned[i] and
                  volume[i] > 1.3 * vol_avg_20[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60) or trend reversal
            if position == 1:
                # Exit long: RSI > 40 or trend turns down (price < EMA)
                if (rsi[i] > 40 or
                    close[i] < ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                # Exit short: RSI < 60 or trend turns up (price > EMA)
                if (rsi[i] < 60 or
                    close[i] > ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_RSI14_4hEMA50_VolumeConfirm_MR"
timeframe = "1h"
leverage = 1.0