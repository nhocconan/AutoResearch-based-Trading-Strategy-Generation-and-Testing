#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(2) extreme reversion with 4h trend filter and volume confirmation
# Long when RSI(2) < 10 AND price > 4h EMA50 (uptrend) AND volume spike (2.0x 20-bar MA)
# Short when RSI(2) > 90 AND price < 4h EMA50 (downtrend) AND volume spike
# RSI(2) captures short-term exhaustion; 4h EMA50 ensures alignment with intermediate trend
# Volume spike confirms institutional participation in the reversal
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend)
# Timeframe: 1h (as required)
# Target: 60-150 total trades over 4 years (15-37/year) to balance signal quality and fee drag

name = "1h_RSI2_4hEMA50_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h RSI(2)
    if len(close) >= 3:
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing for RSI
        def wilders_smoothing(values, period):
            result = np.full_like(values, np.nan)
            if len(values) < period:
                return result
            result[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                result[i] = result[i-1] - (result[i-1] / period) + values[i]
            return result
        
        avg_gain = wilders_smoothing(gain, 2)
        avg_loss = wilders_smoothing(loss, 2)
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_2 = 100 - (100 / (1 + rs))
    else:
        rsi_2 = np.full(n, np.nan)
    
    # Volume confirmation on 1h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_2[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI(2) < 10 (oversold) AND price > 4h EMA50 (uptrend) AND volume spike
            if (rsi_2[i] < 10 and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI(2) > 90 (overbought) AND price < 4h EMA50 (downtrend) AND volume spike
            elif (rsi_2[i] > 90 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI(2) > 50 (mean reversion) OR price < 4h EMA50 (trend break)
            if rsi_2[i] > 50 or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI(2) < 50 (mean reversion) OR price > 4h EMA50 (trend break)
            if rsi_2[i] < 50 or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals