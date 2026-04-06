#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation.
# Long when RSI < 30 and 4h EMA(50) slope > 0 and volume > 1.5x 20-period average.
# Short when RSI > 70 and 4h EMA(50) slope < 0 and volume > 1.5x 20-period average.
# Uses 4h trend to avoid counter-trend trades. Targets 80-160 total trades over 4 years.
# Session filter: 08-20 UTC to avoid low-volume periods.

name = "1h_rsi_meanrev_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA(50) for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_4h_slope = np.diff(ema_4h, prepend=ema_4h[0])
    ema_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_slope)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if not in session
        if not in_session[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Skip if 4h trend data not available
        if np.isnan(ema_4h_slope_aligned[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: RSI > 50 or 4h trend turns bearish
            if (rsi[i] > 50 or 
                ema_4h_slope_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI < 50 or 4h trend turns bullish
            if (rsi[i] < 50 or 
                ema_4h_slope_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and 4h trend filter
            if volume_filter:
                # Long: RSI < 30 and 4h trend bullish
                if (rsi[i] < 30 and 
                    ema_4h_slope_aligned[i] > 0):
                    signals[i] = 0.20
                    position = 1
                # Short: RSI > 70 and 4h trend bearish
                elif (rsi[i] > 70 and 
                      ema_4h_slope_aligned[i] < 0):
                    signals[i] = -0.20
                    position = -1
    
    return signals