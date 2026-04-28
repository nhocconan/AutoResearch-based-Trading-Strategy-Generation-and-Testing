#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d Camarilla pivot R3/S3 fade + volume confirmation
# Long when price breaks above Donchian(20) high AND price < R3(1d) pivot AND volume > 2x avg
# Short when price breaks below Donchian(20) low AND price > S3(1d) pivot AND volume > 2x avg
# Exit when price returns to Donchian midpoint or volume drops
# Target: 12-30 trades/year via tight confluence reducing false breakouts
# Works in bull markets via breakout continuation, in bear via faded breakouts at pivot extremes

name = "6h_Donchian20_1dCamarilla_R3S3_Fade_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for pivot calculation
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily range
    daily_range = high_1d - low_1d
    
    # Camarilla levels (based on previous day's OHLC)
    # R4 = close + range * 1.5
    # R3 = close + range * 1.25
    # R2 = close + range * 1.166
    # R1 = close + range * 1.083
    # PP = (high + low + close) / 3
    # S1 = close - range * 1.083
    # S2 = close - range * 1.166
    # S3 = close - range * 1.25
    # S4 = close - range * 1.5
    camarilla_r3 = close_1d + daily_range * 1.25
    camarilla_s3 = close_1d - daily_range * 1.25
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Prepend NaN for first bar (no previous day data)
    camarilla_r3 = np.concatenate([np.array([np.nan]), camarilla_r3[:-1]])
    camarilla_s3 = np.concatenate([np.array([np.nan]), camarilla_s3[:-1]])
    camarilla_pp = np.concatenate([np.array([np.nan]), camarilla_pp[:-1]])
    
    # Align 1d Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # Calculate Donchian(20) on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: >2.0x 24-bar average volume (4 hours worth on 6h TF)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 24)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        pp = camarilla_pp_aligned[i]
        dh = highest_high[i]  # Donchian high
        dl = lowest_low[i]    # Donchian low
        dm = donchian_mid[i]  # Donchian midpoint
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high AND price < R3 (fade at resistance) AND volume confirmation
            if close[i] > dh and close[i] < r3 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND price > S3 (fade at support) AND volume confirmation
            elif close[i] < dl and close[i] > s3 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price returns to midpoint or volume drops
            if close[i] <= dm or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price returns to midpoint or volume drops
            if close[i] >= dm or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d Camarilla pivot R3/S3 fade + volume confirmation
# Long when price breaks above Donchian(20) high AND price < R3(1d) pivot AND volume > 2x avg
# Short when price breaks below Donchian(20) low AND price > S3(1d) pivot AND volume > 2x avg
# Exit when price returns to Donchian midpoint or volume drops
# Target: 12-30 trades/year via tight confluence reducing false breakouts
# Works in bull markets via breakout continuation, in bear via faded breakouts at pivot extremes

name = "6h_Donchian20_1dCamarilla_R3S3_Fade_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for pivot calculation
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily range
    daily_range = high_1d - low_1d
    
    # Camarilla levels (based on previous day's OHLC)
    # R4 = close + range * 1.5
    # R3 = close + range * 1.25
    # R2 = close + range * 1.166
    # R1 = close + range * 1.083
    # PP = (high + low + close) / 3
    # S1 = close - range * 1.083
    # S2 = close - range * 1.166
    # S3 = close - range * 1.25
    # S4 = close - range * 1.5
    camarilla_r3 = close_1d + daily_range * 1.25
    camarilla_s3 = close_1d - daily_range * 1.25
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Prepend NaN for first bar (no previous day data)
    camarilla_r3 = np.concatenate([np.array([np.nan]), camarilla_r3[:-1]])
    camarilla_s3 = np.concatenate([np.array([np.nan]), camarilla_s3[:-1]])
    camarilla_pp = np.concatenate([np.array([np.nan]), camarilla_pp[:-1]])
    
    # Align 1d Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # Calculate Donchian(20) on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: >2.0x 24-bar average volume (4 hours worth on 6h TF)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 24)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        pp = camarilla_pp_aligned[i]
        dh = highest_high[i]  # Donchian high
        dl = lowest_low[i]    # Donchian low
        dm = donchian_mid[i]  # Donchian midpoint
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high AND price < R3 (fade at resistance) AND volume confirmation
            if close[i] > dh and close[i] < r3 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND price > S3 (fade at support) AND volume confirmation
            elif close[i] < dl and close[i] > s3 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price returns to midpoint or volume drops
            if close[i] <= dm or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price returns to midpoint or volume drops
            if close[i] >= dm or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d Camarilla pivot R3/S3 fade + volume confirmation
# Long when price breaks above Donchian(20) high AND price < R3(1d) pivot AND volume > 2x avg
# Short when price breaks below Donchian(20) low AND price > S3(1d) pivot AND volume > 2x avg
# Exit when price returns to Donchian midpoint or volume drops
# Target: 12-30 trades/year via tight confluence reducing false breakouts
# Works in bull markets via breakout continuation, in bear via faded breakouts at pivot extremes

name = "6h_Donchian20_1dCamarilla_R3S3_Fade_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for pivot calculation
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily range
    daily_range = high_1d - low_1d
    
    # Camarilla levels (based on previous day's OHLC)
    # R4 = close + range * 1.5
    # R3 = close + range * 1.25
    # R2 = close + range * 1.166
    # R1 = close + range * 1.083
    # PP = (high + low + close) / 3
    # S1 = close - range * 1.083
    # S2 = close - range * 1.166
    # S3 = close - range * 1.25
    # S4 = close - range * 1.5
    camarilla_r3 = close_1d + daily_range * 1.25
    camarilla_s3 = close_1d - daily_range * 1.25
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Prepend NaN for first bar (no previous day data)
    camarilla_r3 = np.concatenate([np.array([np.nan]), camarilla_r3[:-1]])
    camarilla_s3 = np.concatenate([np.array([np.nan]), camarilla_s3[:-1]])
    camarilla_pp = np.concatenate([np.array([np.nan]), camarilla_pp[:-1]])
    
    # Align 1d Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # Calculate Donchian(20) on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: >2.0x 24-bar average volume (4 hours worth on 6h TF)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 24)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        pp = camarilla_pp_aligned[i]
        dh = highest_high[i]  # Donchian high
        dl = lowest_low[i]    # Donchian low
        dm = donchian_mid[i]  # Donchian midpoint
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high AND price < R3 (fade at resistance) AND volume confirmation
            if close[i] > dh and close[i] < r3 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND price > S3 (fade at support) AND volume confirmation
            elif close[i] < dl and close[i] > s3 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price returns to midpoint or volume drops
            if close[i] <= dm or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price returns to midpoint or volume drops
            if close[i] >= dm or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d Camarilla pivot R3/S3 fade + volume confirmation
# Long when price breaks above Donchian(20) high AND price < R3(1d) pivot AND volume > 2x avg
# Short when price breaks below Donchian(20) low AND price > S3(1d) pivot AND volume > 2x avg
# Exit when price returns to Donchian midpoint or volume drops
# Target: 12-30 trades/year via tight confluence reducing false breakouts
# Works in bull markets via breakout continuation, in bear via faded breakouts at pivot extremes

name = "6h_Donchian20_1dCamarilla_R3S3_Fade_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for pivot calculation
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily range
    daily_range = high_1d - low_1d
    
    # Camarilla levels (based on previous day's OHLC)
    # R4 = close + range * 1.5
    # R3 = close + range * 1.25
    # R2 = close + range * 1.166
    # R1 = close + range * 1.083
    # PP = (high + low + close) / 3
    # S1 = close - range * 1.083
    # S2 = close - range * 1.166
    # S3 = close - range * 1.25
    # S4 = close - range * 1.5
    camarilla_r3 = close_1d + daily_range * 1.25
    camarilla_s3 = close_1d - daily_range * 1.25
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Prepend NaN for first bar (no previous day data)
    camarilla_r3 = np.concatenate([np.array([np.nan]), camarilla_r3[:-1]])
    camarilla_s3 = np.concatenate([np.array([np.nan]), camarilla_s3[:-1]])
    camarilla_pp = np.concatenate([np.array([np.nan]), camarilla_pp[:-1]])
    
    # Align 1d Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # Calculate Donchian(20) on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: >2.0x 24-bar average volume (4 hours worth on 6h TF)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 24)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        pp = camarilla_pp_aligned[i]
        dh = highest_high[i]  # Donchian high
        dl = lowest_low[i]    # Donchian low
        dm = donchian_mid[i]  # Donchian midpoint
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high AND price < R3 (fade at resistance) AND volume confirmation
            if close[i] > dh and close[i] < r3 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND price > S3 (fade at support) AND volume confirmation
            elif close[i] < dl and close[i] > s3 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price returns to midpoint or volume drops
            if close[i] <= dm or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price returns to midpoint or volume drops
            if close[i] >= dm or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d Camarilla pivot R3/S3 fade + volume confirmation
# Long when price breaks above Donchian(20) high AND price < R3(1d) pivot AND volume > 2x avg
# Short when price breaks below Donchian(20) low AND price > S3(1d) pivot AND volume > 2x avg
# Exit when price returns to Donchian midpoint or volume drops
# Target: 12-30 trades/year via tight confluence reducing false breakouts
# Works in bull markets via breakout continuation, in bear via faded breakouts at pivot extremes

name = "6h_Donchian20_1dCamarilla_R3S3_Fade_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for pivot calculation
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily range
    daily_range = high_1d - low_1d
    
    # Camarilla levels (based on previous day's OHLC)
    # R4 = close + range * 1.5
    # R3 = close + range * 1.25
    # R2 = close + range * 1.166
    # R1 = close + range * 1.083
    # PP = (high + low + close) / 3
    # S1 = close - range * 1.083
    # S2 = close - range * 1.166
    # S3 = close - range * 1.25
    # S4 = close - range * 1.5
    camarilla_r3 = close_1d + daily_range * 1.25
    camarilla_s3 = close_1d - daily_range * 1.25
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Prepend NaN for first bar (no previous day data)
    camarilla_r3 = np.concatenate([np.array([np.nan]), camarilla_r3[:-1]])
    camarilla_s3 = np.concatenate([np.array([np.nan]), camarilla_s3[:-1]])
    camarilla_pp = np.concatenate([np.array([np.nan]), camarilla_pp[:-1]])
    
    # Align 1d Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # Calculate Donchian(20) on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: >2.0x 24-bar average volume (4 hours worth on 6h TF)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 24)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        pp = camarilla_pp_aligned[i]
        dh = highest_high[i]  # Donchian high
        dl = lowest_low[i]    # Donchian low
        dm = donchian_mid[i]  # Donchian midpoint
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high AND price < R3 (fade at resistance) AND volume confirmation
            if close[i] > dh and close[i] < r3 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND price > S3 (fade at support) AND volume confirmation
            elif close[i] < dl and close[i] > s3 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price returns to midpoint or volume drops
            if close[i] <= dm or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price returns to midpoint or volume drops
            if close[i] >= dm or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d Camarilla pivot R3/S3 fade + volume confirmation
# Long when price breaks above Donchian(20) high AND price < R3(1d) pivot AND volume > 2x avg
# Short when price breaks below Donchian(20) low AND price > S3(1d) pivot AND volume > 2x avg
# Exit when price returns to Donchian midpoint or volume drops
# Target: 12-30 trades/year via tight confluence reducing false breakouts
# Works in bull markets via breakout continuation, in bear via faded breakouts at pivot extremes

name = "6h_Donchian20_1dCamarilla_R3S3_Fade_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get