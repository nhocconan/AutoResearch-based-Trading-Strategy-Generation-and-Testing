#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d TK Cross Filter and Volume Spike
# - Uses Ichimoku Cloud (Tenkan/Kijun/Senkou Span A/B) from 6h for trend identification
# - 1d TK Cross (Tenkan-Kijun crossover) as higher timeframe trend filter
# - Volume confirmation: 6h volume > 1.8x 20-period volume SMA
# - Long when: price above cloud, 6h TK bullish cross, 1d TK bullish, volume spike
# - Short when: price below cloud, 6h TK bearish cross, 1d TK bearish, volume spike
# - Exit: price crosses opposite TK line or cloud reversal
# - Position sizing: 0.25 discrete level
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Ichimoku provides dynamic support/resistance, TK cross for momentum, volume for conviction

name = "6h_1d_ichimoku_tk_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Ichimoku components (6h)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    tenkan = (pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
              pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    tenkan = tenkan.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    kijun = (pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max() + 
             pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    kijun = kijun.values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    senkou_b = (pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2
    senkou_b = senkou_b.values
    
    # Calculate 1d TK Cross (Tenkan-Kijun crossover)
    # Tenkan-sen 1d
    tenkan_1d = (pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max() + 
                 pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min()) / 2
    tenkan_1d = tenkan_1d.values
    # Kijun-sen 1d
    kijun_1d = (pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max() + 
                pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min()) / 2
    kijun_1d = kijun_1d.values
    # TK Cross 1d: 1 when bullish (Tenkan > Kijun), -1 when bearish (Tenkan < Kijun)
    tk_cross_1d = np.where(tenkan_1d > kijun_1d, 1, np.where(tenkan_1d < kijun_1d, -1, 0))
    tk_cross_1d_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_1d)
    
    # Volume confirmation: 6h volume > 1.8x 20-period volume SMA
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Track entry condition for exit logic
    entry_tan = np.full(n, np.nan)
    entry_kijun = np.full(n, np.nan)
    
    for i in range(max(period_kijun, period_senkou_b), n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(volume_sma_20[i]) or 
            np.isnan(tk_cross_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_a[i], senkou_b[i])
        lower_cloud = np.minimum(senkou_a[i], senkou_b[i])
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.8 * volume_sma_20[i]
        
        # 6h TK Cross
        tk_bullish_6h = tenkan[i] > kijun[i]
        tk_bearish_6h = tenkan[i] < kijun[i]
        
        # 1d TK bias
        tk_bullish_1d = tk_cross_1d_aligned[i] > 0
        tk_bearish_1d = tk_cross_1d_aligned[i] < 0
        
        if position == 0:  # Flat - look for entry
            # Long: price above cloud, 6h TK bullish, 1d TK bullish, volume spike
            if (close[i] > upper_cloud and tk_bullish_6h and tk_bullish_1d and vol_confirm):
                position = 1
                signals[i] = 0.25
                entry_tan[i] = tenkan[i]
                entry_kijun[i] = kijun[i]
            # Short: price below cloud, 6h TK bearish, 1d TK bearish, volume spike
            elif (close[i] < lower_cloud and tk_bearish_6h and tk_bearish_1d and vol_confirm):
                position = -1
                signals[i] = -0.25
                entry_tan[i] = tenkan[i]
                entry_kijun[i] = kijun[i]
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit: price crosses below Tenkan OR cloud turns bearish (price below cloud)
            exit_condition = (close[i] < tenkan[i]) or (close[i] < lower_cloud)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_tan[i] = np.nan
                entry_kijun[i] = np.nan
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit: price crosses above Tenkan OR cloud turns bullish (price above cloud)
            exit_condition = (close[i] > tenkan[i]) or (close[i] > upper_cloud)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_tan[i] = np.nan
                entry_kijun[i] = np.nan
            else:
                signals[i] = -0.25
    
    return signals