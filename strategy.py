#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation
# Enter long when: RSI(14) < 30, price > 4h EMA(20), volume > 1.5x avg, during active session (08-20 UTC)
# Enter short when: RSI(14) > 70, price < 4h EMA(20), volume > 1.5x avg, during active session
# Exit when RSI returns to neutral zone (40-60)
# Uses 4h trend to filter counter-trend trades, targeting 100-200 trades over 4 years

name = "1h_rsi_meanrev_4h_ema_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) on 1h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 4h EMA(20) for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_20 = pd.Series(close_4h).ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for EMA to stabilize
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_20_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 1:  # long position
            # Exit: RSI > 60 (return to neutral)
            if rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI < 40 (return to neutral)
            if rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: RSI extreme + trend filter + volume + session
            if in_session and volume[i] > volume_threshold[i]:
                if rsi[i] < 30 and close[i] > ema_20_aligned[i]:
                    # Oversold but above 4h EMA - bullish mean reversion
                    signals[i] = 0.20
                    position = 1
                elif rsi[i] > 70 and close[i] < ema_20_aligned[i]:
                    # Overbought but below 4h EMA - bearish mean reversion
                    signals[i] = -0.20
                    position = -1
    
    return signals