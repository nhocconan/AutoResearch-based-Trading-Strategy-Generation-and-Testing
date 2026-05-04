#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA Trend + 1d RSI Filter + Volume Confirmation
# KAMA adapts to market noise - slow in ranging markets, fast in trends.
# Long when KAMA rising AND price above KAMA AND 1d RSI > 50 (bullish bias) AND volume spike.
# Short when KAMA falling AND price below KAMA AND 1d RSI < 50 (bearish bias) AND volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-40 trades/year.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# Volume spike confirms institutional participation. 1d RSI filter avoids counter-trend trades.

name = "4h_KAMA_1dRSI_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF RSI filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI(14) for trend filter
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate KAMA on 4h data
    # Efficiency Ratio (ER) = |change| / sum(|changes|)
    # Smooth Constant (SC) = [ER * (fastest - slowest) + slowest]^2
    # KAMA(prev) + SC * [price - KAMA(prev)]
    close_s = pd.Series(close)
    change = abs(close_s.diff(1))
    volatility = change.rolling(window=10, min_periods=1).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    fastest = 2 / (2 + 2)  # EMA(2)
    slowest = 2 / (30 + 2)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA rising AND price > KAMA AND 1d RSI > 50 AND volume spike
            if (kama[i] > kama[i-1] and 
                close[i] > kama[i] and 
                rsi_1d_aligned[i] > 50 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA falling AND price < KAMA AND 1d RSI < 50 AND volume spike
            elif (kama[i] < kama[i-1] and 
                  close[i] < kama[i] and 
                  rsi_1d_aligned[i] < 50 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA turns down OR price crosses below KAMA OR 1d RSI < 40
            if (kama[i] <= kama[i-1] or 
                close[i] < kama[i] or 
                rsi_1d_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA turns up OR price crosses above KAMA OR 1d RSI > 60
            if (kama[i] >= kama[i-1] or 
                close[i] > kama[i] or 
                rsi_1d_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals