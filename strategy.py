#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4-hour trend filter + 1-hour momentum with volume confirmation
# Uses 4-hour EMA for trend direction (avoids counter-trend trades)
# Enters on 1-hour momentum bursts (RSI + price action) with volume > 1.5x average
# Exits when momentum fades or trend changes
# Target: 20-40 trades/year per symbol (80-160 total over 4 years)
# Session filter: 08-20 UTC to avoid low-liquidity hours

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA(20) for trend direction
    ema_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h RSI(14) for momentum
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # 1h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    # Assuming prices.index is DatetimeIndex from data pipeline
    try:
        hours = prices.index.hour
    except:
        # Fallback if index not DatetimeIndex (shouldn't happen in practice)
        hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # RSI and EMA warmup
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_4h_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Trend filter: price vs 4h EMA
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        # Momentum conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: uptrend + RSI oversold bounce + volume
            if (uptrend and 
                rsi_oversold and 
                volume_spike):
                position = 1
                signals[i] = position_size
            # Enter short: downtrend + RSI overbought bounce + volume
            elif (downtrend and 
                  rsi_overbought and 
                  volume_spike):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend change or momentum fade
            if not uptrend or rsi[i] > 50:  # trend down or RSI recovering
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: trend change or momentum fade
            if not downtrend or rsi[i] < 50:  # trend up or RSI declining
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_EMA_RSI_Momentum_Volume_v1"
timeframe = "1h"
leverage = 1.0