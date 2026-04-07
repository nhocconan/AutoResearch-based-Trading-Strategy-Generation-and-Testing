#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h trend filter and volume filter
# Uses RSI(14) for mean reversion signals:
# - Long when RSI < 30 and price > 4h EMA200 (uptrend filter)
# - Short when RSI > 70 and price < 4h EMA200 (downtrend filter)
# - Volume confirmation: current volume > 20-period average
# - Time filter: only trade 08-20 UTC to avoid low-volume sessions
# - Position size: 0.20 (20% of capital) to limit drawdown
# Designed for low frequency (~20-40 trades/year) to minimize fee drag
# Works in bull/bear via trend filter: only trade in direction of 4h trend

name = "1h_rsi_meanrev_4h_trend_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA200 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # already datetime64[ms], .hour works
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_200_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter from 4h EMA200
        uptrend = close[i] > ema_200_4h_aligned[i]
        downtrend = close[i] < ema_200_4h_aligned[i]
        
        # Exit conditions: reverse signal or trend change
        if position == 1:  # Long position
            if rsi[i] > 50 or not uptrend or not in_session:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
        elif position == -1:  # Short position
            if rsi[i] < 50 or not downtrend or not in_session:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Mean reversion entries with trend and volume confirmation
            if in_session and vol_confirm:
                # Buy on oversold in uptrend
                if rsi[i] < 30 and uptrend:
                    position = 1
                    signals[i] = 0.20
                # Sell on overbought in downtrend
                elif rsi[i] > 70 and downtrend:
                    position = -1
                    signals[i] = -0.20
    
    return signals