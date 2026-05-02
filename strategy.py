#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d ADX regime filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; ADX filters for trending vs ranging markets
# Long when %R crosses above -80 (oversold recovery) + ADX > 25 (trending) + volume spike
# Short when %R crosses below -20 (overbought rejection) + ADX > 25 (trending) + volume spike
# Uses discrete position sizing (0.25) to minimize fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) for 6h timeframe

name = "6h_WilliamsR_1dADX_Trend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend filter
    # ADX requires +DI, -DI, and TR
    tr1 = pd.Series(df_1d['high']).rolling(2).apply(lambda x: x.iloc[1] - x.iloc[0], raw=False)
    tr2 = pd.Series(df_1d['high']).rolling(2).apply(lambda x: abs(x.iloc[1] - df_1d['close'].iloc[x.index[0]-1] if x.index[0] > 0 else 0), raw=False)
    tr3 = pd.Series(df_1d['low']).rolling(2).apply(lambda x: abs(x.iloc[1] - df_1d['close'].iloc[x.index[0]-1] if x.index[0] > 0 else 0), raw=False)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    plus_dm = pd.Series(df_1d['high']).diff()
    minus_dm = pd.Series(df_1d['low']).diff().copy()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)
    
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_1d = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Williams %R(14) on 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    williams_r = williams_r.values
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Williams %R and ADX calculations)
    start_idx = 50  # buffer for 20-period volume MA and 14-period Williams %R/ADX
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 (oversold recovery) + ADX > 25 + volume spike
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                adx_1d_aligned[i] > 25 and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought rejection) + ADX > 25 + volume spike
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  adx_1d_aligned[i] > 25 and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (momentum loss) or ADX < 20 (trend weakening)
            if williams_r[i] < -50 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (momentum loss) or ADX < 20 (trend weakening)
            if williams_r[i] > -50 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals