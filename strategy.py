#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume spike + choppiness regime filter
# Camarilla levels provide high-probability reversal points, volume spike confirms participation
# Choppiness index (14) > 61.8 ensures we only trade in ranging markets where mean reversion works
# Works in bull/bear: chop filter avoids trending markets, volume confirms legitimacy of pivot touches
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_h2 = np.full(n, np.nan)
    camarilla_h1 = np.full(n, np.nan)
    camarilla_l1 = np.full(n, np.nan)
    camarilla_l2 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(n):
        if i < 1:
            # Need previous day's data
            camarilla_h4[i] = np.nan
            camarilla_h3[i] = np.nan
            camarilla_h2[i] = np.nan
            camarilla_h1[i] = np.nan
            camarilla_l1[i] = np.nan
            camarilla_l2[i] = np.nan
            camarilla_l3[i] = np.nan
            camarilla_l4[i] = np.nan
        else:
            # Previous day's OHLC
            phigh = df_1d['high'].iloc[i-1] if i-1 < len(df_1d) else df_1d['high'].iloc[-1]
            plow = df_1d['low'].iloc[i-1] if i-1 < len(df_1d) else df_1d['low'].iloc[-1]
            pclose = df_1d['close'].iloc[i-1] if i-1 < len(df_1d) else df_1d['close'].iloc[-1]
            
            # Camarilla calculations
            rang = phigh - plow
            camarilla_h4[i] = phigh + (rang * 1.1 / 2)
            camarilla_h3[i] = phigh + (rang * 1.1 / 4)
            camarilla_h2[i] = phigh + (rang * 1.1 / 6)
            camarilla_h1[i] = phigh + (rang * 1.1 / 12)
            camarilla_l1[i] = pclose - (rang * 1.1 / 12)
            camarilla_l2[i] = pclose - (rang * 1.1 / 6)
            camarilla_l3[i] = pclose - (rang * 1.1 / 4)
            camarilla_l4[i] = pclose - (rang * 1.1 / 2)
    
    # Calculate 1d Choppiness Index (14-period)
    chop = np.full(n, np.nan)
    if len(df_1d) >= 14:
        tr_1d = np.maximum(df_1d['high'].values, np.roll(df_1d['close'].values, 1))
        tr_1d = np.maximum(tr_1d, np.roll(df_1d['low'].values, 1))
        tr_1d = np.maximum(tr_1d, np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)))
        tr_1d = np.maximum(tr_1d, np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1)))
        tr_1d[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]  # First TR
        
        atr_14 = np.full(len(df_1d), np.nan)
        for i in range(len(df_1d)):
            if i < 13:
                atr_14[i] = np.nan
            else:
                atr_14[i] = np.mean(tr_1d[i-13:i+1])
        
        # Chop = 100 * log10(sum(ATR14) / (max(high)-min(low))) / log10(14)
        for i in range(n):
            idx_1d = min(i, len(df_1d)-1)
            if idx_1d < 13:
                chop[i] = np.nan
            else:
                # Get 14-period high/low range from 1d data
                start_idx = max(0, idx_1d - 13)
                period_high = np.max(df_1d['high'].iloc[start_idx:idx_1d+1])
                period_low = np.min(df_1d['low'].iloc[start_idx:idx_1d+1])
                if atr_14[idx_1d] > 0 and period_high > period_low:
                    chop[i] = 100 * np.log10(atr_14[idx_1d] * 14 / (period_high - period_low)) / np.log10(14)
                else:
                    chop[i] = np.nan
    
    # Align HTF indicators to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 19:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(chop_aligned[i]) or np.isnan(avg_volume[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * avg_volume[i]
        
        # Choppiness regime filter: only trade when chop > 61.8 (ranging market)
        chop_filter = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price < Camarilla L3 OR chop drops below 40 (trending start)
            if close[i] < camarilla_l3_aligned[i] or chop_aligned[i] < 40.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Camarilla H3 OR chop drops below 40 (trending start)
            if close[i] > camarilla_h3_aligned[i] or chop_aligned[i] < 40.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation, chop filter, and Camarilla touch
            if volume_confirmed and chop_filter:
                # Long entry: price touches Camarilla L4 (strong support) with volume
                if abs(close[i] - camarilla_l4_aligned[i]) < (high[i] - low[i]) * 0.3:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches Camarilla H4 (strong resistance) with volume
                elif abs(close[i] - camarilla_h4_aligned[i]) < (high[i] - low[i]) * 0.3:
                    position = -1
                    signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume spike + choppiness regime filter
# Camarilla levels provide high-probability reversal points, volume spike confirms participation
# Choppiness index (14) > 61.8 ensures we only trade in ranging markets where mean reversion works
# Works in bull/bear: chop filter avoids trending markets, volume confirms legitimacy of pivot touches
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_h2 = np.full(n, np.nan)
    camarilla_h1 = np.full(n, np.nan)
    camarilla_l1 = np.full(n, np.nan)
    camarilla_l2 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(n):
        if i < 1:
            # Need previous day's data
            camarilla_h4[i] = np.nan
            camarilla_h3[i] = np.nan
            camarilla_h2[i] = np.nan
            camarilla_h1[i] = np.nan
            camarilla_l1[i] = np.nan
            camarilla_l2[i] = np.nan
            camarilla_l3[i] = np.nan
            camarilla_l4[i] = np.nan
        else:
            # Previous day's OHLC
            phigh = df_1d['high'].iloc[i-1] if i-1 < len(df_1d) else df_1d['high'].iloc[-1]
            plow = df_1d['low'].iloc[i-1] if i-1 < len(df_1d) else df_1d['low'].iloc[-1]
            pclose = df_1d['close'].iloc[i-1] if i-1 < len(df_1d) else df_1d['close'].iloc[-1]
            
            # Camarilla calculations
            rang = phigh - plow
            camarilla_h4[i] = phigh + (rang * 1.1 / 2)
            camarilla_h3[i] = phigh + (rang * 1.1 / 4)
            camarilla_h2[i] = phigh + (rang * 1.1 / 6)
            camarilla_h1[i] = phigh + (rang * 1.1 / 12)
            camarilla_l1[i] = pclose - (rang * 1.1 / 12)
            camarilla_l2[i] = pclose - (rang * 1.1 / 6)
            camarilla_l3[i] = pclose - (rang * 1.1 / 4)
            camarilla_l4[i] = pclose - (rang * 1.1 / 2)
    
    # Calculate 1d Choppiness Index (14-period)
    chop = np.full(n, np.nan)
    if len(df_1d) >= 14:
        tr_1d = np.maximum(df_1d['high'].values, np.roll(df_1d['close'].values, 1))
        tr_1d = np.maximum(tr_1d, np.roll(df_1d['low'].values, 1))
        tr_1d = np.maximum(tr_1d, np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)))
        tr_1d = np.maximum(tr_1d, np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1)))
        tr_1d[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]  # First TR
        
        atr_14 = np.full(len(df_1d), np.nan)
        for i in range(len(df_1d)):
            if i < 13:
                atr_14[i] = np.nan
            else:
                atr_14[i] = np.mean(tr_1d[i-13:i+1])
        
        # Chop = 100 * log10(sum(ATR14) / (max(high)-min(low))) / log10(14)
        for i in range(n):
            idx_1d = min(i, len(df_1d)-1)
            if idx_1d < 13:
                chop[i] = np.nan
            else:
                # Get 14-period high/low range from 1d data
                start_idx = max(0, idx_1d - 13)
                period_high = np.max(df_1d['high'].iloc[start_idx:idx_1d+1])
                period_low = np.min(df_1d['low'].iloc[start_idx:idx_1d+1])
                if atr_14[idx_1d] > 0 and period_high > period_low:
                    chop[i] = 100 * np.log10(atr_14[idx_1d] * 14 / (period_high - period_low)) / np.log10(14)
                else:
                    chop[i] = np.nan
    
    # Align HTF indicators to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 19:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(chop_aligned[i]) or np.isnan(avg_volume[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * avg_volume[i]
        
        # Choppiness regime filter: only trade when chop > 61.8 (ranging market)
        chop_filter = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price < Camarilla L3 OR chop drops below 40 (trending start)
            if close[i] < camarilla_l3_aligned[i] or chop_aligned[i] < 40.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Camarilla H3 OR chop drops below 40 (trending start)
            if close[i] > camarilla_h3_aligned[i] or chop_aligned[i] < 40.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation, chop filter, and Camarilla touch
            if volume_confirmed and chop_filter:
                # Long entry: price touches Camarilla L4 (strong support) with volume
                if abs(close[i] - camarilla_l4_aligned[i]) < (high[i] - low[i]) * 0.3:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches Camarilla H4 (strong resistance) with volume
                elif abs(close[i] - camarilla_h4_aligned[i]) < (high[i] - low[i]) * 0.3:
                    position = -1
                    signals[i] = -0.25
    
    return signals