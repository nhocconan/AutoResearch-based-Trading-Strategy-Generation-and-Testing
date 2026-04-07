#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w RSI trend filter and volume confirmation
# Long when price breaks above 20-day high + weekly RSI > 50 + volume > 20-day average
# Short when price breaks below 20-day low + weekly RSI < 50 + volume > 20-day average
# Exit when price crosses opposite Donchian band or weekly RSI crosses 50
# Stoploss at 2 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses weekly RSI for trend filter to avoid counter-trend trades
# Target: 50-100 total trades over 4 years (12-25/year)

name = "1d_donchian20_1w_rsi_vol_v1"
timeframe = "1d"
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
    
    # 1-week data for RSI trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1-week RSI(14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_s = pd.Series(gain)
    loss_s = pd.Series(loss)
    avg_gain = gain_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # 20-day Donchian channels
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-day average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 14-day ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below Donchian low or weekly RSI < 50
            elif close[i] < low_min[i] or rsi_1w_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above Donchian high or weekly RSI > 50
            elif close[i] > high_max[i] or rsi_1w_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume and weekly RSI filter
            vol_confirm = volume[i] > vol_avg[i]
            
            # Long: price breaks above 20-day high + volume confirmation + weekly RSI > 50
            if close[i] > high_max[i] and vol_confirm and rsi_1w_aligned[i] > 50:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below 20-day low + volume confirmation + weekly RSI < 50
            elif close[i] < low_min[i] and vol_confirm and rsi_1w_aligned[i] < 50:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals