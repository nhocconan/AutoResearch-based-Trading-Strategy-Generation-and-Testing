#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Extreme + 1d Volume Spike + 12h EMA50 Trend Filter
# Williams %R(14) < -80 = oversold (long), > -20 = overbought (short)
# Enter only when 1d volume > 2.0x 20-bar average (volume spike confirms conviction)
# Trend filter: 12h EMA50 slope > 0 for longs, < 0 for shorts (trade with 12h trend)
# Exit when Williams %R returns to -50 (mean reversion) or volume drops below 1.5x average
# Designed for 4h timeframe: targets 20-50 trades/year via tight volume spike + extreme %R conditions
# Works in bull markets (buy oversold in uptrend) and bear markets (sell overbought in downtrend)

name = "4h_WilliamsR_Extreme_1dVolumeSpike_12hEMA50_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate Williams %R(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # Calculate 1d volume spike: >2.0x 20-bar average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > 2.0 * volume_ma_20_1d
    
    # Calculate 12h EMA50 and its slope for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Slope: positive = uptrend, negative = downtrend
    ema_50_slope_12h = np.diff(ema_50_12h, prepend=ema_50_12h[0])
    
    # Prepend zeros for alignment (since we lost bars in calculations)
    williams_r = np.concatenate([np.full(13, np.nan), williams_r])  # 14-1 for rolling
    volume_spike_1d = np.concatenate([np.full(19, np.nan), volume_spike_1d])  # 20-1 for rolling
    ema_50_slope_12h = np.concatenate([np.full(49, np.nan), ema_50_slope_12h])  # 50-1 for EMA + 1 for diff
    
    # Align 1d and 12h indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    ema_50_slope_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_slope_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 20, 50)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(ema_50_slope_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        vol_spike = volume_spike_1d_aligned[i]
        ema_slope = ema_50_slope_12h_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Williams %R < -80 (oversold) AND volume spike AND 12h EMA50 slope > 0 (uptrend)
            if wr < -80 and vol_spike and ema_slope > 0:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R > -20 (overbought) AND volume spike AND 12h EMA50 slope < 0 (downtrend)
            elif wr > -20 and vol_spike and ema_slope < 0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Williams %R >= -50 (mean reversion) or no volume spike
            if wr >= -50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Williams %R <= -50 (mean reversion) or no volume spike
            if wr <= -50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Extreme + 1d Volume Spike + 12h EMA50 Trend Filter
# Williams %R(14) < -80 = oversold (long), > -20 = overbought (short)
# Enter only when 1d volume > 2.0x 20-bar average (volume spike confirms conviction)
# Trend filter: 12h EMA50 slope > 0 for longs, < 0 for shorts (trade with 12h trend)
# Exit when Williams %R returns to -50 (mean reversion) or volume drops below 1.5x average
# Designed for 4h timeframe: targets 20-50 trades/year via tight volume spike + extreme %R conditions
# Works in bull markets (buy oversold in uptrend) and bear markets (sell overbought in downtrend)

name = "4h_WilliamsR_Extreme_1dVolumeSpike_12hEMA50_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate Williams %R(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # Calculate 1d volume spike: >2.0x 20-bar average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > 2.0 * volume_ma_20_1d
    
    # Calculate 12h EMA50 and its slope for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Slope: positive = uptrend, negative = downtrend
    ema_50_slope_12h = np.diff(ema_50_12h, prepend=ema_50_12h[0])
    
    # Prepend zeros for alignment (since we lost bars in calculations)
    williams_r = np.concatenate([np.full(13, np.nan), williams_r])  # 14-1 for rolling
    volume_spike_1d = np.concatenate([np.full(19, np.nan), volume_spike_1d])  # 20-1 for rolling
    ema_50_slope_12h = np.concatenate([np.full(49, np.nan), ema_50_slope_12h])  # 50-1 for EMA + 1 for diff
    
    # Align 1d and 12h indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    ema_50_slope_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_slope_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 20, 50)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(ema_50_slope_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        vol_spike = volume_spike_1d_aligned[i]
        ema_slope = ema_50_slope_12h_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Williams %R < -80 (oversold) AND volume spike AND 12h EMA50 slope > 0 (uptrend)
            if wr < -80 and vol_spike and ema_slope > 0:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R > -20 (overbought) AND volume spike AND 12h EMA50 slope < 0 (downtrend)
            elif wr > -20 and vol_spike and ema_slope < 0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Williams %R >= -50 (mean reversion) or no volume spike
            if wr >= -50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Williams %R <= -50 (mean reversion) or no volume spike
            if wr <= -50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals