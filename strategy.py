#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Camarilla pivots calculated from previous 1d OHLC: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
# Long: Close > R3 AND price > 12h EMA50 AND volume > 2.0x 20-bar avg
# Short: Close < S3 AND price < 12h EMA50 AND volume > 2.0x 20-bar avg
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
# Works in bull via breakout continuation, in bear via mean reversion at extremes (pivots act as support/resistance)

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Need previous 1d candle for Camarilla calculation
        if i < 96:  # 96 * 4h = 384h = 16 days, but we need daily data - using 1d HTF via get_htf_data would be better
            # Instead, we'll use 1d data from mtf_data for Camarilla
            pass
        
        # Load 1d data for Camarilla pivots (using daily timeframe)
        df_1d = get_htf_data(prices, '1d')
        if len(df_1d) < 2:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels from previous 1d candle
        # We need the OHLC of the completed 1d bar that closed before current 4h bar
        # Since we're in 4h timeframe, we can get the last completed 1d bar
        idx_1d = len(df_1d) - 1  # last completed 1d bar
        if idx_1d < 0:
            signals[i] = 0.0
            continue
            
        prev_1d_high = df_1d['high'].iloc[idx_1d]
        prev_1d_low = df_1d['low'].iloc[idx_1d]
        prev_1d_close = df_1d['close'].iloc[idx_1d]
        
        # Camarilla R3 and S3
        rang = prev_1d_high - prev_1d_low
        r3 = prev_1d_close + 1.1 * rang * 1.1 / 4
        s3 = prev_1d_close - 1.1 * rang * 1.1 / 4
        
        curr_close = close[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Close below R3 (breakout failed) OR price below 12h EMA50 (trend change)
            if curr_close < r3 or curr_close < curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Close above S3 (breakdown failed) OR price above 12h EMA50 (trend change)
            if curr_close > s3 or curr_close > curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Close > R3 AND price > 12h EMA50 AND volume spike
            if (curr_close > r3 and 
                curr_close > curr_ema_12h and
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: Close < S3 AND price < 12h EMA50 AND volume spike
            elif (curr_close < s3 and 
                  curr_close < curr_ema_12h and
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Camarilla pivots calculated from previous 1d OHLC: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
# Long: Close > R3 AND price > 12h EMA50 AND volume > 2.0x 20-bar avg
# Short: Close < S3 AND price < 12h EMA50 AND volume > 2.0x 20-bar avg
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
# Works in bull via breakout continuation, in bear via mean reversion at extremes (pivots act as support/resistance)

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Load 1d data for Camarilla pivots (using daily timeframe)
        df_1d = get_htf_data(prices, '1d')
        if len(df_1d) < 2:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla levels from previous 1d candle
        # We need the OHLC of the completed 1d bar that closed before current 4h bar
        # Since we're in 4h timeframe, we can get the last completed 1d bar
        idx_1d = len(df_1d) - 1  # last completed 1d bar
        if idx_1d < 0:
            signals[i] = 0.0
            continue
            
        prev_1d_high = df_1d['high'].iloc[idx_1d]
        prev_1d_low = df_1d['low'].iloc[idx_1d]
        prev_1d_close = df_1d['close'].iloc[idx_1d]
        
        # Camarilla R3 and S3
        rang = prev_1d_high - prev_1d_low
        r3 = prev_1d_close + 1.1 * rang * 1.1 / 4
        s3 = prev_1d_close - 1.1 * rang * 1.1 / 4
        
        curr_close = close[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Close below R3 (breakout failed) OR price below 12h EMA50 (trend change)
            if curr_close < r3 or curr_close < curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Close above S3 (breakdown failed) OR price above 12h EMA50 (trend change)
            if curr_close > s3 or curr_close > curr_ema_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Close > R3 AND price > 12h EMA50 AND volume spike
            if (curr_close > r3 and 
                curr_close > curr_ema_12h and
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: Close < S3 AND price < 12h EMA50 AND volume spike
            elif (curr_close < s3 and 
                  curr_close < curr_ema_12h and
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals