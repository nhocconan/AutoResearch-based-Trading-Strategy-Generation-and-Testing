#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 1d trend filter, volume confirmation, and session filter.
# Uses daily EMA(50) to filter counter-trend trades in strong moves, targeting 75-150 trades over 4 years.
# Long: RSI(14) < 30, price > 1d EMA(50), volume > 1.5x avg, 08-20 UTC
# Short: RSI(14) > 70, price < 1d EMA(50), volume > 1.5x avg, 08-20 UTC
# Exit on RSI returning to neutral (40-60) or price crossing 1d EMA(50) against position.
# Discrete sizing (0.20) to minimize churn, with strict entry conditions to control trade frequency.

name = "1h_rsi_meanrev_1dema_vol_session_v1"
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
    
    # RSI(14) on 1h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], 100)  # Handle division by zero
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for EMA to stabilize
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 1:  # long position
            # Exit: RSI > 60 OR RSI < 30 (deep oversold) OR price < 1d EMA(50)
            if rsi[i] > 60 or rsi[i] < 30 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI < 40 OR RSI > 70 (deep overbought) OR price > 1d EMA(50)
            if rsi[i] < 40 or rsi[i] > 70 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: RSI extreme + trend filter + volume + session
            if in_session and volume[i] > volume_threshold[i]:
                if rsi[i] < 30 and close[i] > ema_50_aligned[i]:
                    # Oversold but above daily EMA - bullish mean reversion
                    signals[i] = 0.20
                    position = 1
                elif rsi[i] > 70 and close[i] < ema_50_aligned[i]:
                    # Overbought but below daily EMA - bearish mean reversion
                    signals[i] = -0.20
                    position = -1
    
    return signals