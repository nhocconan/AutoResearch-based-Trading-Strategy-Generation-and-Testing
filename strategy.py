#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h RSI Pullback with 4h Trend + Volume Spike
# Hypothesis: In strong 4h trends, RSI pullbacks on 1h with volume confirmation provide
# high-probability entries. Works in bull/bear by following 4h trend direction.
# Target: 15-35 trades/year (60-140 total over 4 years).

name = "1h_rsi_pullback_4h_trend_volume_v1"
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
    
    # Get 4h data for trend filter and RSI calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h RSI(14) for overbought/oversold levels
    delta_4h = pd.Series(close_4h).diff()
    gain_4h = delta_4h.where(delta_4h > 0, 0.0)
    loss_4h = -delta_4h.where(delta_4h < 0, 0.0)
    avg_gain_4h = gain_4h.ewm(alpha=1/14, adjust=False).mean()
    avg_loss_4h = loss_4h.ewm(alpha=1/14, adjust=False).mean()
    rs_4h = avg_gain_4h / avg_loss_4h.replace(0, np.nan)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h = rsi_4h.fillna(50).values  # Fill NaN with 50 (neutral)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # 1h RSI(14) for entry signals
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume confirmation: volume > 1.8x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=12).mean().values
    vol_spike = volume > (1.8 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check session and volume confirmation
        if not (session_ok[i] and vol_spike[i]):
            if position != 0:
                # Maintain position but don't add
                signals[i] = 0.20 if position == 1 else -0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses below 40 or trend turns bearish
            if rsi[i] < 40 or close[i] < ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: RSI crosses above 60 or trend turns bullish
            if rsi[i] > 60 or close[i] > ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: RSI pullback from oversold in uptrend
            if (rsi[i] < 30 and rsi_4h_aligned[i] > 50 and 
                close[i] > ema_50_4h_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short: RSI pullback from overbought in downtrend
            elif (rsi[i] > 70 and rsi_4h_aligned[i] < 50 and 
                  close[i] < ema_50_4h_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals