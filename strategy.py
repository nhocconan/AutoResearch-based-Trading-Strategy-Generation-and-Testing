#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and 1d volume regime
# Long when 1h RSI crosses above 50, price > 4h EMA20, and 1d volume > 1.5x 20-day average
# Short when 1h RSI crosses below 50, price < 4h EMA20, and 1d volume > 1.5x 20-day average
# Exit when RSI reaches opposite extreme (70 for long, 30 for short)
# Uses volume regime to filter low-activity periods and 4h EMA for trend alignment
# Target: 60-150 total trades over 4 years (15-37/year) with controlled frequency

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (precomputed)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # 1h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 40  # for RSI and 20-period averages
    
    for i in range(start, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or hours[i] < 8 or hours[i] > 20):
            signals[i] = 0.0
            continue
        
        rsi_now = rsi[i]
        rsi_prev = rsi[i-1]
        price = close[i]
        vol_1d_current = volume[i]
        
        if position == 0:
            # Long setup: RSI crosses above 50, price > 4h EMA20, high volume regime
            if (rsi_prev <= 50 and rsi_now > 50 and
                price > ema_20_4h_aligned[i] and
                vol_1d_current > 1.5 * vol_ma_20_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: RSI crosses below 50, price < 4h EMA20, high volume regime
            elif (rsi_prev >= 50 and rsi_now < 50 and
                  price < ema_20_4h_aligned[i] and
                  vol_1d_current > 1.5 * vol_ma_20_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI reaches 70 (overbought)
            if rsi_now >= 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI reaches 30 (oversold)
            if rsi_now <= 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_RSI50_Cross_4hEMA20_1dVolRegime"
timeframe = "1h"
leverage = 1.0