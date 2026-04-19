#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI with 4h trend filter and volume spike
# - RSI(14) for mean reversion: long when <30, short when >70
# - 4h EMA(50) trend filter: only take longs when price > 4h EMA50, shorts when price < 4h EMA50
# - Volume > 1.5x 20-period average for confirmation
# - Session filter: only trade 08-20 UTC to avoid low-volume hours
# - Exit on opposite RSI extreme or trend reversal
# - Designed for low trade frequency (15-30/year) to minimize fee drag
# - Works in bull/bear by following 4h trend direction

name = "1h_RSI_4hTrend_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA(50) for trend direction
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # RSI(14) on 1h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume filter: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
            
        # Check session: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Volume confirmation
        volume_filter = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0 and in_session:
            # Look for long entry: uptrend + oversold RSI + volume
            if close[i] > ema_50_4h_aligned[i] and rsi[i] < 30 and volume_filter:
                signals[i] = 0.20
                position = 1
            # Look for short entry: downtrend + overbought RSI + volume
            elif close[i] < ema_50_4h_aligned[i] and rsi[i] > 70 and volume_filter:
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long position: exit on overbought RSI or trend reversal
            if rsi[i] > 70 or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short position: exit on oversold RSI or trend reversal
            if rsi[i] < 30 or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals