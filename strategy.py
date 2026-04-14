#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and daily volume spike
# Long when RSI(14) < 30, price > 4h EMA(50), and volume > 2x 24h average
# Short when RSI(14) > 70, price < 4h EMA(50), and volume > 2x 24h average
# Exit when RSI crosses 50
# Uses 4h EMA for trend direction, 1h for entry timing, daily volume for filter
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag

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
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily average volume (24-period)
    vol_daily = df_daily['volume'].values
    vol_ma_daily = pd.Series(vol_daily).rolling(window=24, min_periods=24).mean().values
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align indicators to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 50  # for RSI and EMA calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma_daily_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1h_current = volume[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long setup: RSI oversold, price above 4h EMA, volume spike
            if (rsi_val < 30 and 
                price > ema_4h_aligned[i] and 
                vol_1h_current > 2.0 * vol_ma_daily_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: RSI overbought, price below 4h EMA, volume spike
            elif (rsi_val > 70 and 
                  price < ema_4h_aligned[i] and 
                  vol_1h_current > 2.0 * vol_ma_daily_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses above 50
            if rsi_val > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses below 50
            if rsi_val < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_RSI_MeanReversion_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0