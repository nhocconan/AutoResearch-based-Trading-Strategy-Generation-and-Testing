# 6H_1W_1D_RSI_Trend_Bounce
# Hypothesis: On 6h timeframe, enter long when RSI(6) < 30 and price touches weekly support (1w low) with weekly uptrend (price > weekly EMA50), and volume confirmation.
# Short when RSI(6) > 70 and price touches weekly resistance (1w high) with weekly downtrend (price < weekly EMA50), and volume confirmation.
# Uses weekly trend filter to avoid counter-trend trades and weekly support/resistance for mean reversion entries.
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "6H_1W_1D_RSI_Trend_Bounce"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and support/resistance
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly trend: EMA(50) on close
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema_50_1w
    weekly_downtrend = close_1w < ema_50_1w
    
    # Weekly support/resistance: use weekly low/high as S/R levels
    support_1w = low_1w
    resistance_1w = high_1w
    
    # RSI(6) on 6h closes
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/6, adjust=False, min_periods=6).mean()
    avg_loss = loss.ewm(alpha=1/6, adjust=False, min_periods=6).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Align weekly indicators to 6h
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    support_1w_aligned = align_htf_to_ltf(prices, df_1w, support_1w)
    resistance_1w_aligned = align_htf_to_ltf(prices, df_1w, resistance_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or 
            np.isnan(support_1w_aligned[i]) or np.isnan(resistance_1w_aligned[i]) or 
            np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI < 30, price near weekly support, weekly uptrend, volume confirmation
            if (rsi_values[i] < 30 and 
                low[i] <= support_1w_aligned[i] * 1.005 and  # within 0.5% of support
                weekly_uptrend_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI > 70, price near weekly resistance, weekly downtrend, volume confirmation
            elif (rsi_values[i] > 70 and 
                  high[i] >= resistance_1w_aligned[i] * 0.995 and  # within 0.5% of resistance
                  weekly_downtrend_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 50 (momentum fade) or trend changes or price breaks support
            if (rsi_values[i] > 50 or 
                not weekly_uptrend_aligned[i] or 
                close[i] < support_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI < 50 (momentum fade) or trend changes or price breaks resistance
            if (rsi_values[i] < 50 or 
                not weekly_downtrend_aligned[i] or 
                close[i] > resistance_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals