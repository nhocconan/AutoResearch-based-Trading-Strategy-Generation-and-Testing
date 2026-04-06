#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h ADX trend filter and volume confirmation
# Long when price breaks above 20-period Donchian high AND 12h ADX > 25 AND volume > 1.5x average
# Short when price breaks below 20-period Donchian low AND 12h ADX > 25 AND volume > 1.5x average
# Exit when price crosses Donchian midline (10-period average) or ADX drops below 20
# Uses 4h timeframe to target 75-200 trades over 4 years (19-50/year) with trend-following edge
# Works in both bull/bear markets by following established trends with ADX filter

name = "4h_donchian20_12h_adx_vol_v1"
timeframe = "4h"
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
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # ADX (14-period) from 12h timeframe for trend strength
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.concatenate([[high_12h[0]], high_12h[:-1]])) > 
                       (np.concatenate([[low_12h[0]], low_12h[:-1]]) - low_12h), 
                       np.maximum(high_12h - np.concatenate([[high_12h[0]], high_12h[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[low_12h[0]], low_12h[:-1]]) - low_12h) > 
                        (high_12h - np.concatenate([[high_12h[0]], high_12h[:-1]])), 
                        np.maximum(np.concatenate([[low_12h[0]], low_12h[:-1]]) - low_12h, 0), 0)
    
    # Smooth TR, DM+, DM-
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean()
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean()
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (tr_smooth + 1e-10)
    di_minus = 100 * dm_minus_smooth / (tr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean()
    adx_12h = adx.values
    
    # Align 12h ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses Donchian midline OR ADX drops below 20 (weakening trend)
        if position == 1:  # long position
            if close[i] <= donchian_mid[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_mid[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with strong trend (ADX > 25) and volume confirmation
            # Long: price breaks above Donchian high
            if (close[i] > donchian_high[i] and adx_aligned[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low
            elif (close[i] < donchian_low[i] and adx_aligned[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h ADX trend filter and volume confirmation
# Long when price breaks above 20-period Donchian high AND 12h ADX > 25 AND volume > 1.5x average
# Short when price breaks below 20-period Donchian low AND 12h ADX > 25 AND volume > 1.5x average
# Exit when price crosses Donchian midline (10-period average) or ADX drops below 20
# Uses 4h timeframe to target 75-200 trades over 4 years (19-50/year) with trend-following edge
# Works in both bull/bear markets by following established trends with ADX filter

name = "4h_donchian20_12h_adx_vol_v1"
timeframe = "4h"
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
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # ADX (14-period) from 12h timeframe for trend strength
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.concatenate([[high_12h[0]], high_12h[:-1]])) > 
                       (np.concatenate([[low_12h[0]], low_12h[:-1]]) - low_12h), 
                       np.maximum(high_12h - np.concatenate([[high_12h[0]], high_12h[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[low_12h[0]], low_12h[:-1]]) - low_12h) > 
                        (high_12h - np.concatenate([[high_12h[0]], high_12h[:-1]])), 
                        np.maximum(np.concatenate([[low_12h[0]], low_12h[:-1]]) - low_12h, 0), 0)
    
    # Smooth TR, DM+, DM-
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean()
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean()
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (tr_smooth + 1e-10)
    di_minus = 100 * dm_minus_smooth / (tr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean()
    adx_12h = adx.values
    
    # Align 12h ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses Donchian midline OR ADX drops below 20 (weakening trend)
        if position == 1:  # long position
            if close[i] <= donchian_mid[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_mid[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with strong trend (ADX > 25) and volume confirmation
            # Long: price breaks above Donchian high
            if (close[i] > donchian_high[i] and adx_aligned[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low
            elif (close[i] < donchian_low[i] and adx_aligned[i] > 25 and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals