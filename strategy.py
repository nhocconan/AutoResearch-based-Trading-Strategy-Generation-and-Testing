#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h RSI for trend direction and 1d volume filter for confirmation
# Long when 4h RSI > 55 (bullish) and 1d volume > 1.2x 20-day average
# Short when 4h RSI < 45 (bearish) and 1d volume > 1.2x 20-day average
# Uses 1h only for entry timing (price closes above/below 4h EMA20)
# Target: 15-30 trades per year (60-120 over 4 years) with 0.20 position sizing
# Designed to work in bull markets via RSI > 55 + volume confirmation
# and in bear markets via RSI < 45 + volume confirmation

name = "1h_4hRSI_1dVolume_Trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for RSI calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4h RSI(14)
    delta = pd.Series(df_4h['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align 4h RSI to 1h timeframe (wait for 4h bar close)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_values)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-day average volume
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_filter = df_1d['volume'].values > (1.2 * vol_ma_20)
    
    # Align 1d volume filter to 1h timeframe (wait for 1d bar close)
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    
    # Calculate 4h EMA20 for entry timing
    ema_20 = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after EMA warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_20_aligned[i]) or 
            np.isnan(vol_filter_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: 4h RSI > 55 (bullish), price above 4h EMA20, and volume filter
            if (rsi_4h_aligned[i] > 55 and 
                close[i] > ema_20_aligned[i] and 
                vol_filter_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: 4h RSI < 45 (bearish), price below 4h EMA20, and volume filter
            elif (rsi_4h_aligned[i] < 45 and 
                  close[i] < ema_20_aligned[i] and 
                  vol_filter_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: 4h RSI < 45 (bearish reversal) or price below 4h EMA20
            if (rsi_4h_aligned[i] < 45 or 
                close[i] < ema_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: 4h RSI > 55 (bullish reversal) or price above 4h EMA20
            if (rsi_4h_aligned[i] > 55 or 
                close[i] > ema_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals