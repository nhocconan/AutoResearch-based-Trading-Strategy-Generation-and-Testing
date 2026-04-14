#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and daily volume regime filter
# Long when RSI < 30 and price > 4h EMA50 (bullish trend) and daily volume > 1.5x average
# Short when RSI > 70 and price < 4h EMA50 (bearish trend) and daily volume > 1.5x average
# Exit when RSI crosses back to neutral (40 for long exit, 60 for short exit)
# Uses volume regime filter to avoid low-volume whipsaws
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and cost
# Mean reversion works in ranging markets, trend filter avoids counter-trend trades

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h and daily data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate 1h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_ma_daily = pd.Series(vol_daily).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 50  # for RSI and EMA calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1h_current = volume[i]
        
        if position == 0:
            # Long setup: RSI oversold, price above 4h EMA50, volume spike
            if (rsi[i] < 30 and 
                price > ema_50_4h_aligned[i] and 
                vol_1h_current > 1.5 * vol_ma_daily_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: RSI overbought, price below 4h EMA50, volume spike
            elif (rsi[i] > 70 and 
                  price < ema_50_4h_aligned[i] and 
                  vol_1h_current > 1.5 * vol_ma_daily_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses above 40 (mean reversion complete)
            if rsi[i] > 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses below 60 (mean reversion complete)
            if rsi[i] < 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_RSI_MeanReversion_4hTrend_VolumeRegime"
timeframe = "1h"
leverage = 1.0