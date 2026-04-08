#!/usr/bin/env python3
# 4h_daily_camarilla_pivot_volume_chop_v1
# Hypothesis: 4h breakout at Camarilla pivot levels (H3/L3) with 1d volume spike and choppiness regime filter.
# Long: price > H3 AND 1d volume > 1.5x 20-period average AND chop > 61.8 (ranging market -> mean reversion at extremes)
# Short: price < L3 AND 1d volume > 1.5x 20-period average AND chop > 61.8
# Exit: price crosses H4/L4 levels (intraday mean reversion target) OR chop < 38.2 (trending regime -> follow trend)
# Designed to capture reversals in ranging markets while avoiding whipsaws in strong trends.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-50 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_daily_camarilla_pivot_volume_chop_v1"
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
    
    # 1d Camarilla pivot levels (based on previous 1d candle)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar: based on previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    H3 = pivot + (range_hl * 1.1 / 4)
    L3 = pivot - (range_hl * 1.1 / 4)
    H4 = pivot + (range_hl * 1.1 / 2)
    L4 = pivot - (range_hl * 1.1 / 2)
    
    # Align to 4h timeframe (wait for 1d bar to close)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # 1d volume spike filter
    vol_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ratio = df_1d['volume'].values / vol_ma
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # Choppiness index regime filter (1d)
    # CHOP > 61.8 = ranging (favorable for mean reversion at extremes)
    # CHOP < 38.2 = trending (unfavorable for fade)
    hl1 = pd.DataFrame({'high': df_1d['high'], 'low': df_1d['low'], 'close': df_1d['close']})
    atr1 = pd.Series(np.maximum(np.maximum(hl1['high'] - hl1['low'], 
                                             np.abs(hl1['high'] - hl1['close'].shift(1))),
                                np.abs(hl1['low'] - hl1['close'].shift(1)))).rolling(window=14, min_periods=14).mean()
    abs_close_ch = np.abs(hl1['close'] - hl1['close'].shift(14)).rolling(window=14, min_periods=14).sum()
    chop = 100 * np.log10(abs_close_ch / (atr1 * 14)) / np.log10(10)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any HTF data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(vol_ratio_aligned[i]) or np.isnan(chop_aligned[i])):
            continue
            
        price = close[i]
        
        # Long condition: price breaks above H3 (strong resistance) in ranging market with volume confirmation
        if (price > H3_aligned[i] and 
            vol_ratio_aligned[i] > 1.5 and 
            chop_aligned[i] > 61.8):
            signals[i] = 0.25  # Long 25%
            
        # Short condition: price breaks below L3 (strong support) in ranging market with volume confirmation
        elif (price < L3_aligned[i] and 
              vol_ratio_aligned[i] > 1.5 and 
              chop_aligned[i] > 61.8):
            signals[i] = -0.25  # Short 25%
            
        # Exit conditions: price reaches H4/L4 (mean reversion target) OR chop < 38.2 (trending regime)
        elif (price >= H4_aligned[i] or price <= L4_aligned[i] or chop_aligned[i] < 38.2):
            signals[i] = 0.0  # Flat
            
        # Otherwise hold current signal (handled by np.zeros initialization and persistence)
    
    return signals

#!/usr/bin/env python3
# 4h_daily_camarilla_pivot_volume_chop_v1
# Hypothesis: 4h breakout at Camarilla pivot levels (H3/L3) with 1d volume spike and choppiness regime filter.
# Long: price > H3 AND 1d volume > 1.5x 20-period average AND chop > 61.8 (ranging market -> mean reversion at extremes)
# Short: price < L3 AND 1d volume > 1.5x 20-period average AND chop > 61.8
# Exit: price crosses H4/L4 levels (intraday mean reversion target) OR chop < 38.2 (trending regime -> follow trend)
# Designed to capture reversals in ranging markets while avoiding whipsaws in strong trends.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-50 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_daily_camarilla_pivot_volume_chop_v1"
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
    
    # 1d Camarilla pivot levels (based on previous 1d candle)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar: based on previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    H3 = pivot + (range_hl * 1.1 / 4)
    L3 = pivot - (range_hl * 1.1 / 4)
    H4 = pivot + (range_hl * 1.1 / 2)
    L4 = pivot - (range_hl * 1.1 / 2)
    
    # Align to 4h timeframe (wait for 1d bar to close)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # 1d volume spike filter
    vol_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ratio = df_1d['volume'].values / vol_ma
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # Choppiness index regime filter (1d)
    # CHOP > 61.8 = ranging (favorable for mean reversion at extremes)
    # CHOP < 38.2 = trending (unfavorable for fade)
    hl1 = pd.DataFrame({'high': df_1d['high'], 'low': df_1d['low'], 'close': df_1d['close']})
    atr1 = pd.Series(np.maximum(np.maximum(hl1['high'] - hl1['low'], 
                                             np.abs(hl1['high'] - hl1['close'].shift(1))),
                                np.abs(hl1['low'] - hl1['close'].shift(1)))).rolling(window=14, min_periods=14).mean()
    abs_close_ch = np.abs(hl1['close'] - hl1['close'].shift(14)).rolling(window=14, min_periods=14).sum()
    chop = 100 * np.log10(abs_close_ch / (atr1 * 14)) / np.log10(10)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any HTF data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(vol_ratio_aligned[i]) or np.isnan(chop_aligned[i])):
            continue
            
        price = close[i]
        
        # Long condition: price breaks above H3 (strong resistance) in ranging market with volume confirmation
        if (price > H3_aligned[i] and 
            vol_ratio_aligned[i] > 1.5 and 
            chop_aligned[i] > 61.8):
            signals[i] = 0.25  # Long 25%
            
        # Short condition: price breaks below L3 (strong support) in ranging market with volume confirmation
        elif (price < L3_aligned[i] and 
              vol_ratio_aligned[i] > 1.5 and 
              chop_aligned[i] > 61.8):
            signals[i] = -0.25  # Short 25%
            
        # Exit conditions: price reaches H4/L4 (mean reversion target) OR chop < 38.2 (trending regime)
        elif (price >= H4_aligned[i] or price <= L4_aligned[i] or chop_aligned[i] < 38.2):
            signals[i] = 0.0  # Flat
            
        # Otherwise hold current signal (handled by np.zeros initialization and persistence)
    
    return signals