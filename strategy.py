#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d mean reversion with 1w trend filter and volume confirmation
# Enter long when: RSI(14) < 30, price > 1w EMA(50), volume > 1.5x avg, during active session (08-20 UTC)
# Enter short when: RSI(14) > 70, price < 1w EMA(50), volume > 1.5x avg, during active session
# Exit when RSI returns to neutral zone (40-60) or opposite extreme is reached
# Uses weekly trend to filter counter-trend trades in strong moves, targeting 30-100 trades over 4 years

name = "1d_rsi_meanrev_1wema_vol_session_v1"
timeframe = "1d"
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
    
    # RSI(14) on 1d
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for EMA to stabilize
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 1:  # long position
            # Exit: RSI > 60 OR RSI < 30 (deep oversold) OR price < 1w EMA(50)
            if rsi[i] > 60 or rsi[i] < 30 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: RSI < 40 OR RSI > 70 (deep overbought) OR price > 1w EMA(50)
            if rsi[i] < 40 or rsi[i] > 70 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: RSI extreme + trend filter + volume + session
            if in_session and volume[i] > volume_threshold[i]:
                if rsi[i] < 30 and close[i] > ema_50_aligned[i]:
                    # Oversold but above weekly EMA - bullish mean reversion
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 70 and close[i] < ema_50_aligned[i]:
                    # Overbought but below weekly EMA - bearish mean reversion
                    signals[i] = -0.25
                    position = -1
    
    return signals