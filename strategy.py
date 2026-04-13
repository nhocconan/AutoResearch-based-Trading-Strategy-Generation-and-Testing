#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h primary with 1d HTF - 1d Camarilla pivot levels + 4h volume spike + 4h chop regime filter
    # Uses proven Camarilla structure from daily timeframe for entries, with 4h timing precision
    # Volume spike confirms institutional interest, chop filter avoids ranging markets
    # Target: 20-50 trades over 4 years (5-12/year) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Camarilla: H4 = Close + 1.1*(High-Low)/2, L4 = Close - 1.1*(High-Low)/2
    # H3 = Close + 1.1*(High-Low)/4, L3 = Close - 1.1*(High-Low)/4
    # H2 = Close + 1.1*(High-Low)/6, L2 = Close - 1.1*(High-Low)/6
    # H1 = Close + 1.1*(High-Low)/12, L1 = Close - 1.1*(High-Low)/12
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First bar has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    camarilla_H4 = prev_close + 1.1 * camarilla_range / 2
    camarilla_L4 = prev_close - 1.1 * camarilla_range / 2
    camarilla_H3 = prev_close + 1.1 * camarilla_range / 4
    camarilla_L3 = prev_close - 1.1 * camarilla_range / 4
    camarilla_H2 = prev_close + 1.1 * camarilla_range / 6
    camarilla_L2 = prev_close - 1.1 * camarilla_range / 6
    camarilla_H1 = prev_close + 1.1 * camarilla_range / 12
    camarilla_L1 = prev_close - 1.1 * camarilla_range / 12
    
    # Calculate 4h ATR (10-period) for chop filter
    def calculate_atr(high, low, close, window=10):
        tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - np.roll(close, 1)[1:]))
        tr1 = np.maximum(tr1, np.abs(low[1:] - np.roll(close, 1)[1:]))
        tr = np.concatenate([[np.nan], tr1])
        return pd.Series(tr).rolling(window=window, min_periods=window).mean().values
    
    atr_4h = calculate_atr(high, low, close, window=10)
    atr_ma_20 = pd.Series(atr_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF Camarilla levels to 4h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    camarilla_H2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H2)
    camarilla_L2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L2)
    camarilla_H1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H1)
    camarilla_L1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L1)
    
    # Align 4h indicators
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)  # Using 1d alignment for 4h data
    atr_ma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_H4_aligned[i]) or 
            np.isnan(camarilla_L4_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i]) or
            np.isnan(atr_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * vol_avg_20_aligned[i]
        
        # Chop regime filter: avoid extremely low volatility (choppy markets)
        vol_filter = atr_4h[i] > 0.5 * atr_ma_20_aligned[i]
        
        # Entry conditions: touch Camarilla H4/L4 with volume and volatility
        touch_H4 = abs(close[i] - camarilla_H4_aligned[i]) < 0.001 * close[i]  # Within 0.1%
        touch_L4 = abs(close[i] - camarilla_L4_aligned[i]) < 0.001 * close[i]  # Within 0.1%
        
        enter_long = touch_L4 and volume_confirmed and vol_filter
        enter_short = touch_H4 and volume_confirmed and vol_filter
        
        # Exit conditions: price reaches opposite Camarilla level or midpoint
        exit_long = position == 1 and (close[i] >= camarilla_H4_aligned[i] or close[i] <= camarilla_L4_aligned[i])
        exit_short = position == -1 and (close[i] <= camarilla_L4_aligned[i] or close[i] >= camarilla_H4_aligned[i])
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_touch_volume_vol_filter_v1"
timeframe = "4h"
leverage = 1.0