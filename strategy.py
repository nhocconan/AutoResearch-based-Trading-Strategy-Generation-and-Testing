#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot breakout with daily volume confirmation and 1-day RSI filter
# Long when price closes above H3 Camarilla level AND daily RSI > 50 AND volume > 1.5x 20-period average
# Short when price closes below L3 Camarilla level AND daily RSI < 50 AND volume > 1.5x 20-period average
# Exit when price crosses back to the Camarilla pivot (central level)
# Uses Camarilla pivots for institutional levels, RSI for momentum bias, volume for confirmation
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivots and RSI
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from daily OHLC
    # Pivot = (H + L + C) / 3
    # H3 = Pivot + 1.1 * (H - L) / 2
    # L3 = Pivot - 1.1 * (H - L) / 2
    # H4 = Pivot + 1.1 * (H - L)
    # L4 = Pivot - 1.1 * (H - L)
    daily_pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    daily_range = df_1d['high'] - df_1d['low']
    camarilla_h3 = daily_pivot + 1.1 * daily_range / 2
    camarilla_l3 = daily_pivot - 1.1 * daily_range / 2
    camarilla_pivot = daily_pivot  # central level for exit
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot.values)
    
    # Calculate daily RSI(14) for momentum filter
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: close above H3 AND RSI > 50 AND volume confirmation
            if (price > camarilla_h3_aligned[i] and rsi_1d_aligned[i] > 50 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: close below L3 AND RSI < 50 AND volume confirmation
            elif (price < camarilla_l3_aligned[i] and rsi_1d_aligned[i] < 50 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot level (mean reversion)
            if price <= camarilla_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to pivot level (mean reversion)
            if price >= camarilla_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_H3L3_Volume_RSI"
timeframe = "12h"
leverage = 1.0