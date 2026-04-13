#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h primary with 1d HTF - Camarilla pivot breakout with volume confirmation
    # Designed to capture institutional order flow around key intraday levels with volume validation
    # Target: 100-150 trades over 4 years (25-38/year) for low fee drag and good generalization
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # L4 = C - Range * 1.1/2
    # H4 = C + Range * 1.1/2
    # L3 = C - Range * 1.1/4
    # H3 = C + Range * 1.1/4
    # L2 = C - Range * 1.1/6
    # H2 = C + Range * 1.1/6
    # L1 = C - Range * 1.1/12
    # H1 = C + Range * 1.1/12
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    h4 = close_1d + range_1d * 1.1 / 2
    l4 = close_1d - range_1d * 1.1 / 2
    h3 = close_1d + range_1d * 1.1 / 4
    l3 = close_1d - range_1d * 1.1 / 4
    h2 = close_1d + range_1d * 1.1 / 6
    l2 = close_1d - range_1d * 1.1 / 6
    h1 = close_1d + range_1d * 1.1 / 12
    l1 = close_1d - range_1d * 1.1 / 12
    
    # Calculate 4h ATR (14-period) for volatility filter
    def calculate_atr(high, low, close, window=14):
        tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - np.roll(close, 1)[1:]))
        tr1 = np.maximum(tr1, np.abs(low[1:] - np.roll(close, 1)[1:]))
        tr = np.concatenate([[np.nan], tr1])
        return pd.Series(tr).rolling(window=window, min_periods=window).mean().values
    
    atr_4h = calculate_atr(high, low, close, window=14)
    atr_ma_10 = pd.Series(atr_4h).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 4h primary timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, l2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, l1)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    atr_ma_10_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h2_aligned[i]) or np.isnan(l2_aligned[i]) or
            np.isnan(h1_aligned[i]) or np.isnan(l1_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(atr_ma_10_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * vol_avg_20_aligned[i]
        
        # Volatility filter: avoid extremely low volatility (choppy markets)
        vol_filter = atr_4h[i] > 0.2 * atr_ma_10_aligned[i]
        
        # Breakout conditions at Camarilla levels
        breakout_h4 = close[i] > h4_aligned[i]
        breakdown_l4 = close[i] < l4_aligned[i]
        breakout_h3 = close[i] > h3_aligned[i]
        breakdown_l3 = close[i] < l3_aligned[i]
        
        # Entry conditions: breakout of H3/L3 with volume and volatility confirmation
        enter_long = breakout_h3 and volume_confirmed and vol_filter
        enter_short = breakdown_l3 and volume_confirmed and vol_filter
        
        # Exit conditions: return to H1/L1 levels or opposite H3/L3 break
        exit_long = position == 1 and (close[i] <= h1_aligned[i] or close[i] >= h3_aligned[i])
        exit_short = position == -1 and (close[i] >= l1_aligned[i] or close[i] <= l3_aligned[i])
        
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

name = "4h_1d_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0