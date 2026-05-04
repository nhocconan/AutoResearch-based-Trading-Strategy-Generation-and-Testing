#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI(2) mean reversion + volume confirmation
# Long when KAMA(10,2,30) uptrend, RSI(2) < 10, and volume > 1.5x 20-period EMA
# Short when KAMA downtrend, RSI(2) > 90, and volume > 1.5x 20-period EMA
# Uses 1w EMA50 as regime filter to avoid counter-trend trades in strong weekly trends
# Designed for low trade frequency (target: 15-25/year) to minimize fee drag on 1d timeframe
# Works in bull markets via selective longs in uptrend and bear markets via selective shorts in downtrend

name = "1d_KAMA2_Trend_RSI2_VolumeSpike_1wEMA50_Regime"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF regime filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for regime filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate KAMA(10,2,30) on 1d data
    close_s = pd.Series(close)
    change = abs(close_s.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change.rolling(window=2, min_periods=2).sum() / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    sc = sc.fillna(0)
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(2) on 1d data
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA uptrend, RSI(2) oversold, volume spike, price above weekly EMA50 (bullish regime)
            if (close[i] > kama[i] and 
                rsi[i] < 10 and 
                volume_spike[i] and
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA downtrend, RSI(2) overbought, volume spike, price below weekly EMA50 (bearish regime)
            elif (close[i] < kama[i] and 
                  rsi[i] > 90 and 
                  volume_spike[i] and
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA downtrend OR RSI(2) overbought OR price below weekly EMA50
            if (close[i] < kama[i] or 
                rsi[i] > 70 or
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA uptrend OR RSI(2) oversold OR price above weekly EMA50
            if (close[i] > kama[i] or 
                rsi[i] < 30 or
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals