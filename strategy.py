#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h RSI and 1d volume filter.
# Long: RSI(14) < 30 on 1h + 4h RSI < 40 + volume > 1.5x 20-period avg
# Short: RSI(14) > 70 on 1h + 4h RSI > 60 + volume > 1.5x 20-period avg
# Uses 4h RSI for trend filter to avoid counter-trend trades in strong moves
# Volume filter ensures participation during reversals
# Session filter: 08-20 UTC to avoid low-volume Asian session
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Works in ranging markets and during pullbacks in trends

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-hour RSI for entry signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1h = 100 - (100 / (1 + rs))
    
    # 4-hour RSI for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    delta_4h = np.diff(close_4h, prepend=close_4h[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_4h = avg_gain_4h / (avg_loss_4h + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # 1-day average volume for filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    for i in range(20, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is not ready
        if (np.isnan(rsi_1h[i]) or np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi_1h[i]
        rsi_trend = rsi_4h_aligned[i]
        vol = volume[i]
        avg_vol = avg_volume_1d_aligned[i]
        
        # Volume filter: current volume > 1.5x average volume
        volume_filter = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: RSI oversold + 4h RSI not overbought + volume
            if (rsi_val < 30 and 
                rsi_trend < 40 and
                volume_filter):
                position = 1
                signals[i] = position_size
            # Short: RSI overbought + 4h RSI not oversold + volume
            elif (rsi_val > 70 and 
                  rsi_trend > 60 and
                  volume_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral or 4h RSI turns bearish
            if (rsi_val > 50 or
                rsi_trend > 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral or 4h RSI turns bullish
            if (rsi_val < 50 or
                rsi_trend < 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_RSI_MeanReversion"
timeframe = "1h"
leverage = 1.0