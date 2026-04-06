#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above Donchian upper (20-period) AND 1d ADX > 25 AND volume > 1.5x 20-period average
# Short when price breaks below Donchian lower (20-period) AND 1d ADX > 25 AND volume > 1.5x 20-period average
# Exit when price crosses Donchian midline (10-period average of upper/lower)
# Uses 12h timeframe to reduce trade frequency, 1d ADX for trend strength, volume for confirmation
# Target: 50-150 total trades over 4 years (12-37/year) for optimal 12h performance

name = "12h_donchian20_1d_adx_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # 1-day ADX(14) trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate ADX components
    tr1 = pd.Series(daily_high) - pd.Series(daily_low)
    tr2 = np.abs(pd.Series(daily_high) - pd.Series(daily_close).shift(1))
    tr3 = np.abs(pd.Series(daily_low) - pd.Series(daily_close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    plus_dm = np.where((pd.Series(daily_high) - pd.Series(daily_high).shift(1)) > 
                       (pd.Series(daily_low).shift(1) - pd.Series(daily_low)), 
                       np.maximum(pd.Series(daily_high) - pd.Series(daily_high).shift(1), 0), 0)
    minus_dm = np.where((pd.Series(daily_low).shift(1) - pd.Series(daily_low)) > 
                        (pd.Series(daily_high) - pd.Series(daily_high).shift(1)), 
                        np.maximum(pd.Series(daily_low).shift(1) - pd.Series(daily_low), 0), 0)
    
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    adx_values = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price crosses Donchian midline
        if position == 1:  # long position
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: price breaks above Donchian upper AND ADX > 25 AND volume confirmation
            if (close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1] and 
                adx_1d_aligned[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND ADX > 25 AND volume confirmation
            elif (close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1] and 
                  adx_1d_aligned[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above Donchian upper (20-period) AND 1d ADX > 25 AND volume > 1.5x 20-period average
# Short when price breaks below Donchian lower (20-period) AND 1d ADX > 25 AND volume > 1.5x 20-period average
# Exit when price crosses Donchian midline (10-period average of upper/lower)
# Uses 12h timeframe to reduce trade frequency, 1d ADX for trend strength, volume for confirmation
# Target: 50-150 total trades over 4 years (12-37/year) for optimal 12h performance

name = "12h_donchian20_1d_adx_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # 1-day ADX(14) trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate ADX components
    tr1 = pd.Series(daily_high) - pd.Series(daily_low)
    tr2 = np.abs(pd.Series(daily_high) - pd.Series(daily_close).shift(1))
    tr3 = np.abs(pd.Series(daily_low) - pd.Series(daily_close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    plus_dm = np.where((pd.Series(daily_high) - pd.Series(daily_high).shift(1)) > 
                       (pd.Series(daily_low).shift(1) - pd.Series(daily_low)), 
                       np.maximum(pd.Series(daily_high) - pd.Series(daily_high).shift(1), 0), 0)
    minus_dm = np.where((pd.Series(daily_low).shift(1) - pd.Series(daily_low)) > 
                        (pd.Series(daily_high) - pd.Series(daily_high).shift(1)), 
                        np.maximum(pd.Series(daily_low).shift(1) - pd.Series(daily_low), 0), 0)
    
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    adx_values = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price crosses Donchian midline
        if position == 1:  # long position
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: price breaks above Donchian upper AND ADX > 25 AND volume confirmation
            if (close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1] and 
                adx_1d_aligned[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND ADX > 25 AND volume confirmation
            elif (close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1] and 
                  adx_1d_aligned[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals