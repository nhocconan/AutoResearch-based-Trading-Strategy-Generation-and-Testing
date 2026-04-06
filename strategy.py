#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power (Bull/Bear) with 1d trend filter and volume confirmation
# Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when Bull Power > 0 AND Bear Power < 0 AND 1d EMA50 > EMA200 (uptrend) AND volume > 1.5x avg
# Short when Bear Power > 0 AND Bull Power < 0 AND 1d EMA50 < EMA200 (downtrend) AND volume > 1.5x avg
# Exit when power signals reverse or trend fails
# Targets 50-150 trades over 4 years by requiring trend alignment and volume confirmation
# Works in bull/bear by only taking trend-aligned trades, avoiding counter-trend whipsaws

name = "6h_elder_ray_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray Power (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # 1d trend filter: EMA50 vs EMA200
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    ema200_1d = pd.Series(daily_close).ewm(span=200, adjust=False).mean().values
    
    # Align 1d EMAs to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for EMAs
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            # Exit if bull power fails, bear power takes over, or trend breaks
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or 
                ema50_1d_aligned[i] <= ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit if bear power fails, bull power takes over, or trend breaks
            if (bear_power[i] <= 0 or bull_power[i] >= 0 or 
                ema50_1d_aligned[i] >= ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend alignment and volume confirmation
            # Long: Bull power positive, bear power negative, uptrend, volume confirmation
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                ema50_1d_aligned[i] > ema200_1d_aligned[i] and
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear power positive, bull power negative, downtrend, volume confirmation
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  ema50_1d_aligned[i] < ema200_1d_aligned[i] and
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power (Bull/Bear) with 1d trend filter and volume confirmation
# Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when Bull Power > 0 AND Bear Power < 0 AND 1d EMA50 > EMA200 (uptrend) AND volume > 1.5x avg
# Short when Bear Power > 0 AND Bull Power < 0 AND 1d EMA50 < EMA200 (downtrend) AND volume > 1.5x avg
# Exit when power signals reverse or trend fails
# Targets 50-150 trades over 4 years by requiring trend alignment and volume confirmation
# Works in bull/bear by only taking trend-aligned trades, avoiding counter-trend whipsaws

name = "6h_elder_ray_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray Power (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # 1d trend filter: EMA50 vs EMA200
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    ema200_1d = pd.Series(daily_close).ewm(span=200, adjust=False).mean().values
    
    # Align 1d EMAs to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for EMAs
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            # Exit if bull power fails, bear power takes over, or trend breaks
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or 
                ema50_1d_aligned[i] <= ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit if bear power fails, bull power takes over, or trend breaks
            if (bear_power[i] <= 0 or bull_power[i] >= 0 or 
                ema50_1d_aligned[i] >= ema200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend alignment and volume confirmation
            # Long: Bull power positive, bear power negative, uptrend, volume confirmation
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                ema50_1d_aligned[i] > ema200_1d_aligned[i] and
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear power positive, bull power negative, downtrend, volume confirmation
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  ema50_1d_aligned[i] < ema200_1d_aligned[i] and
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals