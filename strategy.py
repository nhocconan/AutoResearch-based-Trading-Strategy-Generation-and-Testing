#!/usr/bin/env python3
# 6h_donchian_volume_chop_v1
# Hypothesis: 6h Donchian channel breakout with volume confirmation and chop regime filter.
# In trending markets (CHOP < 38.2), we trade breakouts in direction of trend.
# In choppy markets (CHOP > 61.8), we fade extremes at Donchian bands.
# Volume confirmation (>1.5x 20-period average) filters false breakouts.
# Weekly trend filter (price above/below weekly 20 EMA) avoids counter-trend trades.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 12-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_volume_chop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for chop regime and weekly trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # 1d Chop regime (14-period)
    atr_1d = pd.Series(high_d - low_d).rolling(window=14, min_periods=14).mean().values
    highest_high_1d = pd.Series(high_d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_d).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_1d - lowest_low_1d
    chop_denom_safe = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop_1d = 100 * np.log10(atr_1d * np.sqrt(14) / chop_denom_safe) / np.log10(14)
    
    # 1d Weekly EMA for trend filter (using daily close as proxy)
    ema_20_1d = pd.Series(close_d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d data to 6h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # 6h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(ema_20_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        chop = chop_aligned[i]
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches Donchian midpoint or volume dries up
            if close[i] <= donchian_mid[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches Donchian midpoint or volume dries up
            if close[i] >= donchian_mid[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Trending regime: CHOP < 38.2 -> trade breakouts
                if chop < 38.2:
                    # Long breakout: price breaks above upper band with volume AND trend filter
                    if close[i] > highest_high[i] and close[i] > ema_20_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short breakdown: price breaks below lower band with volume AND trend filter
                    elif close[i] < lowest_low[i] and close[i] < ema_20_aligned[i]:
                        position = -1
                        signals[i] = -0.25
                # Choppy regime: CHOP > 61.8 -> fade extremes
                elif chop > 61.8:
                    # Long fade: price touches lower band and reverses up with volume
                    if close[i] <= lowest_low[i] * 1.001 and close[i] > open_prices[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short fade: price touches upper band and reverses down with volume
                    elif close[i] >= highest_high[i] * 0.999 and close[i] < open_prices[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals

# Fix: need to access open prices for reversal detection
open_prices = prices['open'].values if 'open' in prices.columns else close

# Re-defined function with proper open access
def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_prices = prices['open'].values
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for chop regime and weekly trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # 1d Chop regime (14-period)
    atr_1d = pd.Series(high_d - low_d).rolling(window=14, min_periods=14).mean().values
    highest_high_1d = pd.Series(high_d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_d).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_1d - lowest_low_1d
    chop_denom_safe = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop_1d = 100 * np.log10(atr_1d * np.sqrt(14) / chop_denom_safe) / np.log10(14)
    
    # 1d Weekly EMA for trend filter (using daily close as proxy)
    ema_20_1d = pd.Series(close_d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d data to 6h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # 6h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(ema_20_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        chop = chop_aligned[i]
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches Donchian midpoint or volume dries up
            if close[i] <= donchian_mid[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches Donchian midpoint or volume dries up
            if close[i] >= donchian_mid[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Trending regime: CHOP < 38.2 -> trade breakouts
                if chop < 38.2:
                    # Long breakout: price breaks above upper band with volume AND trend filter
                    if close[i] > highest_high[i] and close[i] > ema_20_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short breakdown: price breaks below lower band with volume AND trend filter
                    elif close[i] < lowest_low[i] and close[i] < ema_20_aligned[i]:
                        position = -1
                        signals[i] = -0.25
                # Choppy regime: CHOP > 61.8 -> fade extremes
                elif chop > 61.8:
                    # Long fade: price touches lower band and reverses up with volume
                    if close[i] <= lowest_low[i] * 1.001 and close[i] > open_prices[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short fade: price touches upper band and reverses down with volume
                    elif close[i] >= highest_high[i] * 0.999 and close[i] < open_prices[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
# 6h_donchian_volume_chop_v1
# Hypothesis: 6h Donchian channel breakout with volume confirmation and chop regime filter.
# In trending markets (CHOP < 38.2), we trade breakouts in direction of trend.
# In choppy markets (CHOP > 61.8), we fade extremes at Donchian bands.
# Volume confirmation (>1.5x 20-period average) filters false breakouts.
# Weekly trend filter (price above/below weekly 20 EMA) avoids counter-trend trades.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 12-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_volume_chop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_prices = prices['open'].values
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for chop regime and weekly trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # 1d Chop regime (14-period)
    atr_1d = pd.Series(high_d - low_d).rolling(window=14, min_periods=14).mean().values
    highest_high_1d = pd.Series(high_d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_d).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_1d - lowest_low_1d
    chop_denom_safe = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop_1d = 100 * np.log10(atr_1d * np.sqrt(14) / chop_denom_safe) / np.log10(14)
    
    # 1d Weekly EMA for trend filter (using daily close as proxy)
    ema_20_1d = pd.Series(close_d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d data to 6h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # 6h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(ema_20_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        chop = chop_aligned[i]
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches Donchian midpoint or volume dries up
            if close[i] <= donchian_mid[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches Donchian midpoint or volume dries up
            if close[i] >= donchian_mid[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Trending regime: CHOP < 38.2 -> trade breakouts
                if chop < 38.2:
                    # Long breakout: price breaks above upper band with volume AND trend filter
                    if close[i] > highest_high[i] and close[i] > ema_20_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short breakdown: price breaks below lower band with volume AND trend filter
                    elif close[i] < lowest_low[i] and close[i] < ema_20_aligned[i]:
                        position = -1
                        signals[i] = -0.25
                # Choppy regime: CHOP > 61.8 -> fade extremes
                elif chop > 61.8:
                    # Long fade: price touches lower band and reverses up with volume
                    if close[i] <= lowest_low[i] * 1.001 and close[i] > open_prices[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short fade: price touches upper band and reverses down with volume
                    elif close[i] >= highest_high[i] * 0.999 and close[i] < open_prices[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals