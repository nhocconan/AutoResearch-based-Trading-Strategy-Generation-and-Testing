#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot with 1d trend filter and volume confirmation
# Long when price breaks above Camarilla R4 AND 1d EMA(50) is rising AND volume > 1.5x average
# Short when price breaks below Camarilla S4 AND 1d EMA(50) is falling AND volume > 1.5x average
# Exit when price crosses back through Camarilla H4/L4 (mean reversion) or opposite breakout
# Camarilla levels from 1d provide precise support/resistance; 1d EMA ensures higher timeframe trend alignment; volume confirms institutional interest
# Designed to work in both bull and bear markets by following the dominant trend on 1d timeframe
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from 1d (using previous day's OHLC)
    # Camarilla formulas:
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # H3 = Close + 1.125 * (High - Low)
    # L3 = Close - 1.125 * (High - Low)
    # H2 = Close + 0.75 * (High - Low)
    # L2 = Close - 0.75 * (High - Low)
    # H1 = Close + 0.5 * (High - Low)
    # L1 = Close - 0.5 * (High - Low)
    
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate levels
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)
    camarilla_h3 = prev_close + 1.125 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.125 * (prev_high - prev_low)
    camarilla_h2 = prev_close + 0.75 * (prev_high - prev_low)
    camarilla_l2 = prev_close - 0.75 * (prev_high - prev_low)
    camarilla_h1 = prev_close + 0.5 * (prev_high - prev_low)
    camarilla_l1 = prev_close - 0.5 * (prev_high - prev_low)
    
    # Calculate EMA on 1d (50-period) for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Align 1d data to 6h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (need 50 for EMA, 1 for shift)
    start = 60
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        ema_val = ema_50_aligned[i]
        ema_prev = ema_50_aligned[i-1]
        
        if position == 0:
            # Long setup: price breaks above Camarilla R4 AND 1d EMA rising AND volume confirmation
            if (high_val > camarilla_h4_aligned[i] and ema_val > ema_prev and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below Camarilla S4 AND 1d EMA falling AND volume confirmation
            elif (low_val < camarilla_l4_aligned[i] and ema_val < ema_prev and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Camarilla H3 OR opposite breakout
            if (close_val < camarilla_h3_aligned[i] or 
                low_val < camarilla_l4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Camarilla L3 OR opposite breakout
            if (close_val > camarilla_l3_aligned[i] or 
                high_val > camarilla_h4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Camarilla_1dEMA_Volume"
timeframe = "6h"
leverage = 1.0