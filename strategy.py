#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme with 1d ADX Regime Filter and Volume Spike
# Targets 12-37 trades per year (50-150 total over 4 years) to minimize fee drag
# Williams %R identifies overbought/oversold extremes (>80 = oversold, <20 = overbought)
# 1d ADX > 25 confirms trending market (avoid whipsaw in ranging markets)
# Volume spike (2.0x 20-period average) confirms institutional participation
# Uses discrete position sizing 0.25 to balance exposure and risk
# Works in both bull and bear: ADX regime filter avoids false signals in chop, volume confirms validity

name = "6h_WilliamsR_Extreme_1dADX_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Williams %R and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for Williams %R and ADX calculation
        return np.zeros(n)
    
    # Calculate 1d Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    highest_high = high_1d.rolling(window=14, min_periods=14).max()
    lowest_low = low_1d.rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r = williams_r.values  # Convert to numpy array
    
    # Calculate 1d ADX(14)
    # ADX calculation: +DM, -DM, TR, then smoothed, then DX, then ADX
    plus_dm = high_1d.diff()
    minus_dm = low_1d.diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_1d - low_1d
    tr2 = (high_1d - close_1d.shift()).abs()
    tr3 = (low_1d - close_1d.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
    
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    adx_values = adx.values
    
    # Williams %R signals: < -80 = oversold (long signal), > -20 = overbought (short signal)
    williams_oversold = williams_r < -80
    williams_overbought = williams_r > -20
    
    # ADX regime filter: ADX > 25 = trending market
    adx_trending = adx_values > 25
    
    # Align 1d indicators to 6h timeframe
    williams_oversold_aligned = align_htf_to_ltf(prices, df_1d, williams_oversold.astype(float))
    williams_overbought_aligned = align_htf_to_ltf(prices, df_1d, williams_overbought.astype(float))
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending.astype(float))
    
    # Calculate 6h volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_oversold_aligned[i]) or np.isnan(williams_overbought_aligned[i]) or 
            np.isnan(adx_trending_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80) AND ADX trending (> 25) AND volume spike
            if (williams_oversold_aligned[i] and 
                adx_trending_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND ADX trending (> 25) AND volume spike
            elif (williams_overbought_aligned[i] and 
                  adx_trending_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns above -50 (middle) OR ADX loses trend OR volume normalizes
            if (williams_r_aligned[i] > -50 or 
                not adx_trending_aligned[i] or 
                not volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns below -50 (middle) OR ADX loses trend OR volume normalizes
            if (williams_r_aligned[i] < -50 or 
                not adx_trending_aligned[i] or 
                not volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Note: williams_r_aligned is not defined in the exit conditions above.
# Let me fix this by computing the aligned Williams %R values.
    
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme with 1d ADX Regime Filter and Volume Spike
# Targets 12-37 trades per year (50-150 total over 4 years) to minimize fee drag
# Williams %R identifies overbought/oversold extremes (>80 = oversold, <20 = overbought)
# 1d ADX > 25 confirms trending market (avoid whipsaw in ranging markets)
# Volume spike (2.0x 20-period average) confirms institutional participation
# Uses discrete position sizing 0.25 to balance exposure and risk
# Works in both bull and bear: ADX regime filter avoids false signals in chop, volume confirms validity

name = "6h_WilliamsR_Extreme_1dADX_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Williams %R and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for Williams %R and ADX calculation
        return np.zeros(n)
    
    # Calculate 1d Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    highest_high = high_1d.rolling(window=14, min_periods=14).max()
    lowest_low = low_1d.rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r = williams_r.values  # Convert to numpy array
    
    # Calculate 1d ADX(14)
    # ADX calculation: +DM, -DM, TR, then smoothed, then DX, then ADX
    plus_dm = high_1d.diff()
    minus_dm = low_1d.diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_1d - low_1d
    tr2 = (high_1d - close_1d.shift()).abs()
    tr3 = (low_1d - close_1d.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
    
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    adx_values = adx.values
    
    # Williams %R signals: < -80 = oversold (long signal), > -20 = overbought (short signal)
    williams_oversold = williams_r < -80
    williams_overbought = williams_r > -20
    
    # ADX regime filter: ADX > 25 = trending market
    adx_trending = adx_values > 25
    
    # Align 1d indicators to 6h timeframe
    williams_oversold_aligned = align_htf_to_ltf(prices, df_1d, williams_oversold.astype(float))
    williams_overbought_aligned = align_htf_to_ltf(prices, df_1d, williams_overbought.astype(float))
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending.astype(float))
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)  # For exit conditions
    
    # Calculate 6h volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_oversold_aligned[i]) or np.isnan(williams_overbought_aligned[i]) or 
            np.isnan(adx_trending_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80) AND ADX trending (> 25) AND volume spike
            if (williams_oversold_aligned[i] and 
                adx_trending_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND ADX trending (> 25) AND volume spike
            elif (williams_overbought_aligned[i] and 
                  adx_trending_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns above -50 (middle) OR ADX loses trend OR volume normalizes
            if (williams_r_aligned[i] > -50 or 
                not adx_trending_aligned[i] or 
                not volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns below -50 (middle) OR ADX loses trend OR volume normalizes
            if (williams_r_aligned[i] < -50 or 
                not adx_trending_aligned[i] or 
                not volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals