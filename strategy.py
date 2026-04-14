# hypothesis: 6h donchian breakout with 1d volatility filter and volume confirmation
# long when price breaks above 20-period donchian high AND daily atr > 20-period average atr AND volume > 1.5x 20-period average volume
# short when price breaks below 20-period donchian low AND daily atr > 20-period average atr AND volume > 1.5x 20-period average volume
# exit when price crosses back inside the donchian channel
# targets breakouts during volatile periods with volume confirmation to avoid false breakouts
# targets 15-30 trades per year per symbol

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # load 1d data once for volatility filter
    df_1d = get_htf_data(prices, '1d')
    
    # calculate donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # calculate atr (14-period) for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # calculate daily atr for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1_1d = np.abs(high_1d - low_1d)
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_1d[0] = tr1_1d[0]
    tr3_1d[0] = tr1_1d[0]
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_avg = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_1d_avg_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_avg)
    
    # calculate volume average for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    # start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # skip if any critical data is nan
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_1d_avg_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # long setup: break above donchian high + volatility filter + volume confirmation
            if (price > donchian_high[i] and 
                atr[i] > atr_1d_avg_aligned[i] and 
                vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # short setup: break below donchian low + volatility filter + volume confirmation
            elif (price < donchian_low[i] and 
                  atr[i] > atr_1d_avg_aligned[i] and 
                  vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # exit long: price crosses back inside donchian channel (below donchian low)
            if price < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # exit short: price crosses back inside donchian channel (above donchian high)
            if price > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Donchian_Volatility_Volume"
timeframe = "6h"
leverage = 1.0