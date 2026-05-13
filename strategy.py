#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 12h ADX trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold), price > 12h EMA50 (uptrend), and volume > 1.5x 20-bar average.
# Short when Williams %R > -20 (overbought), price < 12h EMA50 (downtrend), and volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 6h timeframe.
# Williams %R identifies overextended moves; 12h EMA50 ensures higher timeframe trend alignment;
# volume confirmation avoids low-momentum breakouts. Designed for range-bound bear markets like 2025.

name = "6h_WilliamsR_ADX_EMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R (14-period) on 6h data
    lookback_wr = 14
    highest_high = pd.Series(high).rolling(window=lookback_wr, min_periods=lookback_wr).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_wr, min_periods=lookback_wr).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback_wr, lookback_vol) + 1
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Oversold (%R < -80), uptrend (close > 12h EMA50), volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Overbought (%R > -20), downtrend (close < 12h EMA50), volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Overbought (%R > -50) OR trend break (close < 12h EMA50)
            if (williams_r[i] > -50 or 
                close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Oversold (%R < -50) OR trend break (close > 12h EMA50)
            if (williams_r[i] < -50 or 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 12h ADX trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold), price > 12h EMA50 (uptrend), and volume > 1.5x 20-bar average.
# Short when Williams %R > -20 (overbought), price < 12h EMA50 (downtrend), and volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 6h timeframe.
# Williams %R identifies overextended moves; 12h EMA50 ensures higher timeframe trend alignment;
# volume confirmation avoids low-momentum breakouts. Designed for range-bound bear markets like 2025.

name = "6h_WilliamsR_ADX_EMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R (14-period) on 6h data
    lookback_wr = 14
    highest_high = pd.Series(high).rolling(window=lookback_wr, min_periods=lookback_wr).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_wr, min_periods=lookback_wr).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback_wr, lookback_vol) + 1
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Oversold (%R < -80), uptrend (close > 12h EMA50), volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Overbought (%R > -20), downtrend (close < 12h EMA50), volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Overbought (%R > -50) OR trend break (close < 12h EMA50)
            if (williams_r[i] > -50 or 
                close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Oversold (%R < -50) OR trend break (close > 12h EMA50)
            if (williams_r[i] < -50 or 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals