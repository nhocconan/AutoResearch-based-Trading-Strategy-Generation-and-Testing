#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation
# Uses 4h RSI trend to filter 1h mean reversion entries:
# - Long when 1h RSI < 30 AND 4h RSI > 50 (uptrend) AND volume > 20-period average
# - Short when 1h RSI > 70 AND 4h RSI < 50 (downtrend) AND volume > 20-period average
# - Exit when 1h RSI crosses back to neutral zone (40-60)
# - Designed for low frequency (target: 15-30 trades/year) to minimize fee drag
# - Mean reversion works in ranging markets; trend filter avoids counter-trend trades

name = "1h_rsi_meanrev_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h RSI trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate RSI for 4h
    delta_4h = np.diff(close_4h, prepend=close_4h[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_4h = avg_gain_4h / (avg_loss_4h + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # 1h RSI for mean reversion
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter from 4h RSI
        uptrend = rsi_4h_aligned[i] > 50
        downtrend = rsi_4h_aligned[i] < 50
        
        # Mean reversion conditions from 1h RSI
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = (rsi[i] >= 40) & (rsi[i] <= 60)
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit when RSI returns to neutral
            if rsi_neutral[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
        elif position == -1:  # Short position
            # Exit when RSI returns to neutral
            if rsi_neutral[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Entry conditions with trend and volume confirmation
            # Long on RSI oversold in uptrend
            if rsi_oversold and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.20
            # Short on RSI overbought in downtrend
            elif rsi_overbought and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.20
    
    return signals