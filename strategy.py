#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d Camarilla pivot reversal strategy
# - Uses 1d Camarilla pivot levels (R1, S1) for reversal entries
# - Enters long when price crosses above S1 in uptrend (1d EMA34 filter)
# - Enters short when price crosses below R1 in downtrend (1d EMA34 filter)
# - Requires volume confirmation: current 12h volume > 1.5x 20-period average
# - Uses 12h RSI(14) for entry timing: long when RSI < 40, short when RSI > 60
# - Exits on opposite RSI threshold or trend reversal
# - Designed for low-frequency, high-conviction trades in both bull and bear markets
# - Target: 20-40 trades/year to minimize fee drag

name = "12h_Camarilla_Pivot_Reversal_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels
    # Formula: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = (H+L+CLOSE)/3 (typical price)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    hl_range = df_1d['high'] - df_1d['low']
    r1 = typical_price + hl_range * 1.1 / 12
    s1 = typical_price - hl_range * 1.1 / 12
    
    # Align 1d R1 and S1 to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12h volume average (20-period)
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h RSI(14) for entry timing
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_12h[i]) or 
            np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = vol_ma_12h[i] > 0 and volume[i] > 1.5 * vol_ma_12h[i]
        
        if position == 0:
            # Look for long entry: uptrend (price > 1d EMA34) + price > S1 + RSI < 40 + volume
            if (close[i] > ema_34_1d_aligned[i] and 
                close[i] > s1_aligned[i] and 
                rsi_values[i] < 40 and 
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (price < 1d EMA34) + price < R1 + RSI > 60 + volume
            elif (close[i] < ema_34_1d_aligned[i] and 
                  close[i] < r1_aligned[i] and 
                  rsi_values[i] > 60 and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on RSI > 60 or trend reversal or price < S1
            if (rsi_values[i] > 60 or 
                close[i] < ema_34_1d_aligned[i] or 
                close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on RSI < 40 or trend reversal or price > R1
            if (rsi_values[i] < 40 or 
                close[i] > ema_34_1d_aligned[i] or 
                close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals