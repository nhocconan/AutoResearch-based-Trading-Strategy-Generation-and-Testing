#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above R3 AND close > 4h EMA50 AND volume > 1.5x 20-bar avg
# Short when price breaks below S3 AND close < 4h EMA50 AND volume > 1.5x 20-bar avg
# Exits when price reverts to Camarilla pivot (PP) or volume drops
# Target: 15-37 trades/year via tight Camarilla breakout conditions + 4h trend filter
# Uses 4h/1d for signal direction (trend + volatility regime), 1h only for entry timing precision
# Camarilla levels provide structured support/resistance that works in ranging and trending markets
# Volume confirmation reduces false breakouts, trend filter avoids counter-trend trades

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h close
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Camarilla levels from previous bar's OHLC
    # Camarilla: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # We need previous bar's data to avoid look-ahead
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    pp = (prev_high + prev_low + prev_close) / 3.0
    r3 = prev_close + (prev_high - prev_low) * 1.1 / 2.0
    s3 = prev_close - (prev_high - prev_low) * 1.1 / 2.0
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(pp[i]) or np.isnan(r3[i]) or np.isnan(s3[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        in_session = session_filter[i]
        ema_50 = ema_50_4h_aligned[i]
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above R3 AND close > 4h EMA50 AND volume confirmation AND in session
            if price > r3[i] and ema_50 > 0 and vol_conf and in_session:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below S3 AND close < 4h EMA50 AND volume confirmation AND in session
            elif price < s3[i] and ema_50 > 0 and vol_conf and in_session:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price reverts to PP or volume drops or out of session
            if price < pp[i] or not vol_conf or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit when price reverts to PP or volume drops or out of session
            if price > pp[i] or not vol_conf or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals