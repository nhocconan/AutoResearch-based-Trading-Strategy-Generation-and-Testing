#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h ADX trend filter + volume confirmation
# Long when price breaks above Donchian(20) high AND ADX > 25 (trending) AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND ADX > 25 AND volume > 1.5x average
# Exit when price returns to Donchian middle OR ADX < 20 (trend weakening)
# Uses 12h ADX to filter out false breakouts in ranging markets, targeting 75-200 total trades over 4 years
# Works in bull markets via long breakouts and bear markets via short breakouts

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
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_channel = (highest_high + lowest_low) / 2
    
    # ADX (14-period) from 12h timeframe for trend strength
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.append([close_12h[0]], close_12h[:-1]))
    tr3 = np.abs(low_12h - np.append([close_12h[0]], close_12h[:-1]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.append([high_12h[0]], high_12h[:-1])) > 
                       (np.append([low_12h[0]], low_12h[:-1]) - low_12h), 
                       np.maximum(high_12h - np.append([high_12h[0]], high_12h[:-1]), 0), 0)
    dm_minus = np.where((np.append([low_12h[0]], low_12h[:-1]) - low_12h) > 
                        (high_12h - np.append([high_12h[0]], high_12h[:-1])), 
                        np.maximum(np.append([low_12h[0]], low_12h[:-1]) - low_12h, 0), 0)
    
    # Smoothed values
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr_12h + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_12h + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    # Align 12h ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price returns to middle OR trend weakens
        if position == 1:  # long position
            if close[i] <= middle_channel[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= middle_channel[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts in trending market (ADX > 25) with volume confirmation
            # Long breakout: price above upper Donchian band
            if (adx_aligned[i] > 25 and close[i] > highest_high[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price below lower Donchian band
            elif (adx_aligned[i] > 25 and close[i] < lowest_low[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h ADX trend filter + volume confirmation
# Long when price breaks above Donchian(20) high AND ADX > 25 (trending) AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND ADX > 25 AND volume > 1.5x average
# Exit when price returns to Donchian middle OR ADX < 20 (trend weakening)
# Uses 12h ADX to filter out false breakouts in ranging markets, targeting 75-200 total trades over 4 years
# Works in bull markets via long breakouts and bear markets via short breakouts

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
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_channel = (highest_high + lowest_low) / 2
    
    # ADX (14-period) from 12h timeframe for trend strength
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.append([close_12h[0]], close_12h[:-1]))
    tr3 = np.abs(low_12h - np.append([close_12h[0]], close_12h[:-1]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.append([high_12h[0]], high_12h[:-1])) > 
                       (np.append([low_12h[0]], low_12h[:-1]) - low_12h), 
                       np.maximum(high_12h - np.append([high_12h[0]], high_12h[:-1]), 0), 0)
    dm_minus = np.where((np.append([low_12h[0]], low_12h[:-1]) - low_12h) > 
                        (high_12h - np.append([high_12h[0]], high_12h[:-1])), 
                        np.maximum(np.append([low_12h[0]], low_12h[:-1]) - low_12h, 0), 0)
    
    # Smoothed values
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr_12h + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_12h + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    
    # Align 12h ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price returns to middle OR trend weakens
        if position == 1:  # long position
            if close[i] <= middle_channel[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= middle_channel[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts in trending market (ADX > 25) with volume confirmation
            # Long breakout: price above upper Donchian band
            if (adx_aligned[i] > 25 and close[i] > highest_high[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price below lower Donchian band
            elif (adx_aligned[i] > 25 and close[i] < lowest_low[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals