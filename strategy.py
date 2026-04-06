#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index with 1-day Elder Ray trend filter and volume confirmation
# Long when 6h Bull Power > 0 and Bear Power < 0 + 1d Bull Power > 0 + volume > 1.5x average
# Short when 6h Bear Power < 0 and Bull Power > 0 + 1d Bear Power < 0 + volume > 1.5x average
# Exit when 6h Elder Ray signal reverses or volume drops below threshold
# Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# Elder Ray measures bull/bear power relative to EMA, effective in trending and ranging markets

name = "6h_elder_ray_1d_filter_vol_v1"
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
    
    # Calculate EMA(13) for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray components for 6h
    bull_power_6h = high - ema13
    bear_power_6h = low - ema13
    
    # 1-day EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, min_periods=13, adjust=False).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # 1-day Elder Ray components
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if required data not available
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: Elder Ray signal reversal
        if position == 1:  # long position
            if bull_power_6h[i] <= 0 or bear_power_6h[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if bear_power_6h[i] >= 0 or bull_power_6h[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for Elder Ray signals with 1-day trend filter and volume
            # Bullish: 6h bull power positive, bear power negative + 1d bull power positive + volume
            if (bull_power_6h[i] > 0 and bear_power_6h[i] < 0 and 
                bull_power_1d_aligned[i] > 0 and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Bearish: 6h bear power negative, bull power positive + 1d bear power negative + volume
            elif (bear_power_6h[i] < 0 and bull_power_6h[i] > 0 and 
                  bear_power_1d_aligned[i] < 0 and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R with 1-day trend filter and volume confirmation
# Long when Williams %R crosses above -80 (oversold) + 1-day trend up + volume > 1.5x average
# Short when Williams %R crosses below -20 (overbought) + 1-day trend down + volume > 1.5x average
# Exit when Williams %R returns to -50 level or trend reverses
# Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# Williams %R identifies overbought/oversold levels, effective in ranging markets with trend filter

name = "6h_williams_r_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # 1-day EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema21_1d = pd.Series(close_1d).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(williams_r[i]) or np.isnan(ema21_1d_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: Williams %R returns to -50 or trend reversal
        if position == 1:  # long position
            if williams_r[i] >= -50 or close[i] < ema21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if williams_r[i] <= -50 or close[i] > ema21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for Williams %R signals with trend filter and volume
            # Bullish: Williams %R crosses above -80 (oversold) + trend up + volume
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema21_1d_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Bearish: Williams %R crosses below -20 (overbought) + trend down + volume
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema21_1d_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian Channel breakout with 1-day ADX filter and volume confirmation
# Long when price breaks above Donchian(20) high + ADX > 25 + volume > 1.5x average
# Short when price breaks below Donchian(20) low + ADX > 25 + volume > 1.5x average
# Exit when price returns to Donchian middle or ADX drops below 20
# Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# Donchian breakouts capture trends, ADX filters for trending markets only

name = "6h_donchian_1d_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # 1-day ADX (14-period) for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price returns to Donchian middle or ADX weak
        if position == 1:  # long position
            if close[i] <= donch_mid[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donch_mid[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for Donchian breakouts with ADX filter and volume
            # Bullish breakout: price above Donchian high + ADX > 25 + volume
            if (close[i] > donch_high[i] and 
                adx_aligned[i] > 25 and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Bearish breakout: price below Donchian low + ADX > 25 + volume
            elif (close[i] < donch_low[i] and 
                  adx_aligned[i] > 25 and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index with 1-day Elder Ray trend filter and volume confirmation
# Long when 6h Bull Power > 0 and Bear Power < 0 + 1d Bull Power > 0 + volume > 1.5x average
# Short when 6h Bear Power < 0 and Bull Power > 0 + 1d Bear Power < 0 + volume > 1.5x average
# Exit when 6h Elder Ray signal reverses or volume drops below threshold
# Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# Elder Ray measures bull/bear power relative to EMA, effective in trending and ranging markets

name = "6h_elder_ray_1d_filter_vol_v1"
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
    
    # Calculate EMA(13) for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray components for 6h
    bull_power_6h = high - ema13
    bear_power_6h = low - ema13
    
    # 1-day EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, min_periods=13, adjust=False).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # 1-day Elder Ray components
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if required data not available
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: Elder Ray signal reversal
        if position == 1:  # long position
            if bull_power_6h[i] <= 0 or bear_power_6h[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if bear_power_6h[i] >= 0 or bull_power_6h[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for Elder Ray signals with 1-day trend filter and volume
            # Bullish: 6h bull power positive, bear power negative + 1d bull power positive + volume
            if (bull_power_6h[i] > 0 and bear_power_6h[i] < 0 and 
                bull_power_1d_aligned[i] > 0 and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Bearish: 6h bear power negative, bull power positive + 1d bear power negative + volume
            elif (bear_power_6h[i] < 0 and bull_power_6h[i] > 0 and 
                  bear_power_1d_aligned[i] < 0 and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R with 1-day trend filter and volume confirmation
# Long when Williams %R crosses above -80 (oversold) + 1-day trend up + volume > 1.5x average
# Short when Williams %R crosses below -20 (overbought) + 1-day trend down + volume > 1.5x average
# Exit when Williams %R returns to -50 level or trend reverses
# Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# Williams %R identifies overbought/oversold levels, effective in ranging markets with trend filter

name = "6h_williams_r_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # 1-day EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema21_1d = pd.Series(close_1d).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(williams_r[i]) or np.isnan(ema21_1d_aligned[i]) or 
            np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: Williams %R returns to -50 or trend reversal
        if position == 1:  # long position
            if williams_r[i] >= -50 or close[i] < ema21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if williams_r[i] <= -50 or close[i] > ema21_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for Williams %R signals with trend filter and volume
            # Bullish: Williams %R crosses above -80 (oversold) + trend up + volume
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema21_1d_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Bearish: Williams %R crosses below -20 (overbought) + trend down + volume
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema21_1d_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian Channel breakout with 1-day ADX filter and volume confirmation
# Long when price breaks above Donchian(20) high + ADX > 25 + volume > 1.5x average
# Short when price breaks below Donchian(20) low + ADX > 25 + volume > 1.5x average
# Exit when price returns to Donchian middle or ADX drops below 20
# Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# Donchian breakouts capture trends, ADX filters for trending markets only

name = "6h_donchian_1d_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # 1-day ADX (14-period) for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price returns to Donchian middle or ADX weak
        if position == 1:  # long position
            if close[i] <= donch_mid[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donch_mid[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for Donchian breakouts with ADX filter and volume
            # Bullish breakout: price above Donchian high + ADX > 25 + volume
            if (close[i] > donch_high[i] and 
                adx_aligned[i] > 25 and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Bearish breakout: price below Donchian low + ADX > 25 + volume
            elif (close[i] < donch_low[i] and 
                  adx_aligned[i] > 25 and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian Channel breakout with 1-day ADX filter and volume confirmation
# Long when price breaks above Donchian(20) high + ADX > 25 + volume > 1.5x average
# Short when price breaks below Donchian(20) low + ADX > 25 + volume > 1.5x average
# Exit when price returns to Donchian middle or ADX drops below 20
# Uses 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# Donchian breakouts capture trends, ADX filters for trending markets only

name = "6h_donchian_1d_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # 1-day ADX (14-period) for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1,