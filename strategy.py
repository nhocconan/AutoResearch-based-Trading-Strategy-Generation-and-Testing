#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d EMA34 Trend + Volume Spike
# Williams %R identifies overbought/oversold conditions on 6h timeframe
# Extreme readings below -90 (oversold) or above -10 (overbought) signal potential reversals
# 1d EMA34 provides trend filter: only take longs when price > EMA34 in uptrend, shorts when price < EMA34 in downtrend
# Volume spike (2.0x 20-period average) confirms conviction behind the move
# Works in bull markets by buying oversold dips in uptrend and selling overbought rallies in downtrend
# Works in bear markets by selling bounces in downtrend and covering shorts on oversold spikes
# Targets 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_WilliamsR_Extreme_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data ONCE before loop for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    # Calculate 6h Williams %R(14)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close_6h) / (highest_high - lowest_low)) * -100
    
    # Align Williams %R to 6h timeframe (wait for completed 6h bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Load 1d data ONCE before loop for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34)
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 6h timeframe (wait for completed 1d bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate volume spike (2.0x 20-period average) - using 6h volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Williams %R and EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -90 (oversold) + price > 1d EMA34 + volume spike
            if williams_r_aligned[i] < -90 and close[i] > ema_34_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -10 (overbought) + price < 1d EMA34 + volume spike
            elif williams_r_aligned[i] > -10 and close[i] < ema_34_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -10 (overbought) or price < 1d EMA34
            if williams_r_aligned[i] > -10 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -90 (oversold) or price > 1d EMA34
            if williams_r_aligned[i] < -90 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d EMA34 Trend + Volume Spike
# Williams %R identifies overbought/oversold conditions on 6h timeframe
# Extreme readings below -90 (oversold) or above -10 (overbought) signal potential reversals
# 1d EMA34 provides trend filter: only take longs when price > EMA34 in uptrend, shorts when price < EMA34 in downtrend
# Volume spike (2.0x 20-period average) confirms conviction behind the move
# Works in bull markets by buying oversold dips in uptrend and selling overbought rallies in downtrend
# Works in bear markets by selling bounces in downtrend and covering shorts on oversold spikes
# Targets 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_WilliamsR_Extreme_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data ONCE before loop for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    # Calculate 6h Williams %R(14)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close_6h) / (highest_high - lowest_low)) * -100
    
    # Align Williams %R to 6h timeframe (wait for completed 6h bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Load 1d data ONCE before loop for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34)
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 6h timeframe (wait for completed 1d bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate volume spike (2.0x 20-period average) - using 6h volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Williams %R and EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -90 (oversold) + price > 1d EMA34 + volume spike
            if williams_r_aligned[i] < -90 and close[i] > ema_34_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -10 (overbought) + price < 1d EMA34 + volume spike
            elif williams_r_aligned[i] > -10 and close[i] < ema_34_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -10 (overbought) or price < 1d EMA34
            if williams_r_aligned[i] > -10 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -90 (oversold) or price > 1d EMA34
            if williams_r_aligned[i] < -90 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals