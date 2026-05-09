#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume_Mod"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_open = np.roll(prices['open'].values, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    prev_open[0] = prices['open'].values[0]
    
    # Calculate Camarilla R1 and S1 levels
    range_ = prev_high - prev_low
    close_prev = prev_close
    r1 = close_prev + range_ * 1.1 / 6
    s1 = close_prev - range_ * 1.1 / 6
    
    # Daily trend: EMA34 on 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.3x 20-period SMA
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.3 * vol_ma20
    
    # Momentum filter: RSI(14) to avoid overbought/oversold extremes
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    rsi_filter = (rsi > 30) & (rsi < 70)  # Only trade in neutral RSI range
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(r1[i]) or np.isnan(s1[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: breakout above R1 with daily uptrend, volume, and neutral RSI
            if (price > r1[i] and 
                price > ema34_1d_aligned[i] and 
                vol_filter[i] and 
                rsi_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: breakdown below S1 with daily downtrend, volume, and neutral RSI
            elif (price < s1[i] and 
                  price < ema34_1d_aligned[i] and 
                  vol_filter[i] and 
                  rsi_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price returns to daily EMA or loses volume or RSI overbought
            if (price < ema34_1d_aligned[i] or 
                not vol_filter[i] or 
                rsi[i] >= 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to daily EMA or loses volume or RSI oversold
            if (price > ema34_1d_aligned[i] or 
                not vol_filter[i] or 
                rsi[i] <= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals