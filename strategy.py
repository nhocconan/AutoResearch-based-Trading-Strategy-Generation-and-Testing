#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot reversal with 1-day trend filter and volume confirmation
# Long when price touches Camarilla L3 (support) in 1-day uptrend with volume spike
# Short when price touches Camarilla H3 (resistance) in 1-day downtrend with volume spike
# Exit when price reaches opposite Camarilla level (H3 for longs, L3 for shorts) or stoploss at 2 * ATR
# Volume confirmation: current volume > 2.0 * average volume of last 20 periods
# Position size: 0.25 (25% of capital)
# Target: 80-180 total trades over 4 years (20-45/year)

name = "4h_camarilla_reversal_1d_trend_vol_v1"
timeframe = "4h"
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
    
    # 1-day data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1-day EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1-day bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # H3 = close + (high - low) * 1.1/2
    # L3 = close - (high - low) * 1.1/2
    # H4 = close + (high - low) * 1.1
    # L4 = close - (high - low) * 1.1
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d_prev + range_1d * 1.1 / 2
    camarilla_l3 = close_1d_prev - range_1d * 1.1 / 2
    camarilla_h4 = close_1d_prev + range_1d * 1.1
    camarilla_l4 = close_1d_prev - range_1d * 1.1
    
    # Align Camarilla levels to 4h timeframe (using previous day's values)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Average volume for volume confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches Camarilla H3 (take profit) or H4 (stop reversal)
            elif close[i] >= camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches Camarilla L3 (take profit) or L4 (stop reversal)
            elif close[i] <= camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Trend filter: 1-day EMA(50) for uptrend/downtrend
            uptrend = ema_50_1d_aligned[i] > close[i]  # price above EMA = uptrend
            downtrend = ema_50_1d_aligned[i] < close[i]  # price below EMA = downtrend
            
            # Volume confirmation: current volume > 2.0 * average volume
            volume_confirm = volume[i] > 2.0 * vol_avg[i]
            
            # Long: price touches Camarilla L3 in uptrend with volume
            if close[i] <= camarilla_l3_aligned[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price touches Camarilla H3 in downtrend with volume
            elif close[i] >= camarilla_h3_aligned[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals