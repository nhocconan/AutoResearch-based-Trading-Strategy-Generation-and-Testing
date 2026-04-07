#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h RSI Pullback + 4h EMA Trend + Daily Volume Spike
# Hypothesis: RSI pullbacks in the direction of 4h trend with daily volume confirmation
# provide high-probability entries in both bull and bear markets.
# 4h EMA establishes trend, daily volume confirms institutional interest,
# 1h RSI provides precise entry timing during pullbacks.
# Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_rsi_pullback_4h_trend_daily_volume_v1"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h EMA(20) for trend
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d volume average
    vol_avg_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Daily volume confirmation: volume > 1.5x 20-day average
        vol_ok = volume[i] > (1.5 * vol_avg_1d_aligned[i])
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or trend turns bearish
            if rsi[i] > 70 or close[i] < ema_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or trend turns bullish
            if rsi[i] < 30 or close[i] > ema_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if vol_ok:
                # Long: RSI < 30 (oversold) and price above 4h EMA (uptrend)
                if rsi[i] < 30 and close[i] > ema_20_4h_aligned[i]:
                    # Additional confirmation: RSI turning up from oversold
                    if i == 20 or rsi[i] > rsi[i-1]:
                        position = 1
                        signals[i] = 0.20
                # Short: RSI > 70 (overbought) and price below 4h EMA (downtrend)
                elif rsi[i] > 70 and close[i] < ema_20_4h_aligned[i]:
                    # Additional confirmation: RSI turning down from overbought
                    if i == 20 or rsi[i] < rsi[i-1]:
                        position = -1
                        signals[i] = -0.20
    
    return signals