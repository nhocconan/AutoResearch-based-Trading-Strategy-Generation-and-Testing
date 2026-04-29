#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation
# Long when price breaks above R3 AND price > 1d EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below S3 AND price < 1d EMA50 AND volume > 2.0x 20-bar avg
# Exit when price crosses opposite Camarilla level (S3 for longs, R3 for shorts)
# Uses discrete position sizing (0.25) to minimize fee churn while capturing moves.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h.
# Camarilla R3/S3 are strong intraday levels with good breakout reliability.
# 1d EMA50 filters counter-trend moves, volume spike ensures institutional participation.
# Works in bull markets (trend continuation via breakouts) and bear markets (mean reversion within trend via exits).

name = "6h_Camarilla_R3S3_Breakout_1dEMA50_VolumeSpike_v1"
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
    
    # Get 1d data for EMA50 trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get prior bar's OHLC for Camarilla pivot calculation (using completed prior 1d bar)
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Extract prior bar's OHLC (1d timeframe) - shift by 1 to use only completed prior bar
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close = np.roll(df_1d['close'].values, 1)
    # Set first value to NaN as we don't have prior bar
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Align prior bar OHLC to 6h timeframe
    prior_high_aligned = align_htf_to_ltf(prices, df_1d, prior_high)
    prior_low_aligned = align_htf_to_ltf(prices, df_1d, prior_low)
    prior_close_aligned = align_htf_to_ltf(prices, df_1d, prior_close)
    
    # Calculate Camarilla levels for each 6h bar based on prior bar's OHLC
    # Camarilla R3 = Close + (High - Low) * 1.1/4
    # Camarilla S3 = Close - (High - Low) * 1.1/4
    range_hl = prior_high_aligned - prior_low_aligned
    r3 = prior_close_aligned + range_hl * 1.1 / 4
    s3 = prior_close_aligned - range_hl * 1.1 / 4
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 1) + 1  # EMA50 warmup + 1 for prior bar shift
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_1d_aligned[i]
        
        # Camarilla levels
        r3_level = r3[i]
        s3_level = s3[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below S3 (mean reversion to median)
            if curr_close < s3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (mean reversion to median)
            if curr_close > r3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above R3 AND price > 1d EMA50 AND volume confirmation
            if curr_close > r3_level and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND price < 1d EMA50 AND volume confirmation
            elif curr_close < s3_level and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation
# Long when price breaks above R3 AND price > 1d EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below S3 AND price < 1d EMA50 AND volume > 2.0x 20-bar avg
# Exit when price crosses opposite Camarilla level (S3 for longs, R3 for shorts)
# Uses discrete position sizing (0.25) to minimize fee churn while capturing moves.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h.
# Camarilla R3/S3 are strong intraday levels with good breakout reliability.
# 1d EMA50 filters counter-trend moves, volume spike ensures institutional participation.
# Works in bull markets (trend continuation via breakouts) and bear markets (mean reversion within trend via exits).

name = "6h_Camarilla_R3S3_Breakout_1dEMA50_VolumeSpike_v1"
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
    
    # Get 1d data for EMA50 trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get prior bar's OHLC for Camarilla pivot calculation (using completed prior 1d bar)
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Extract prior bar's OHLC (1d timeframe) - shift by 1 to use only completed prior bar
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close = np.roll(df_1d['close'].values, 1)
    # Set first value to NaN as we don't have prior bar
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Align prior bar OHLC to 6h timeframe
    prior_high_aligned = align_htf_to_ltf(prices, df_1d, prior_high)
    prior_low_aligned = align_htf_to_ltf(prices, df_1d, prior_low)
    prior_close_aligned = align_htf_to_ltf(prices, df_1d, prior_close)
    
    # Calculate Camarilla levels for each 6h bar based on prior bar's OHLC
    # Camarilla R3 = Close + (High - Low) * 1.1/4
    # Camarilla S3 = Close - (High - Low) * 1.1/4
    range_hl = prior_high_aligned - prior_low_aligned
    r3 = prior_close_aligned + range_hl * 1.1 / 4
    s3 = prior_close_aligned - range_hl * 1.1 / 4
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 1) + 1  # EMA50 warmup + 1 for prior bar shift
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_1d_aligned[i]
        
        # Camarilla levels
        r3_level = r3[i]
        s3_level = s3[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below S3 (mean reversion to median)
            if curr_close < s3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (mean reversion to median)
            if curr_close > r3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above R3 AND price > 1d EMA50 AND volume confirmation
            if curr_close > r3_level and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND price < 1d EMA50 AND volume confirmation
            elif curr_close < s3_level and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation
# Long when price breaks above R3 AND price > 1d EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below S3 AND price < 1d EMA50 AND volume > 2.0x 20-bar avg
# Exit when price crosses opposite Camarilla level (S3 for longs, R3 for shorts)
# Uses discrete position sizing (0.25) to minimize fee churn while capturing moves.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h.
# Camarilla R3/S3 are strong intraday levels with good breakout reliability.
# 1d EMA50 filters counter-trend moves, volume spike ensures institutional participation.
# Works in bull markets (trend continuation via breakouts) and bear markets (mean reversion within trend via exits).

name = "6h_Camarilla_R3S3_Breakout_1dEMA50_VolumeSpike_v1"
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
    
    # Get 1d data for EMA50 trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get prior bar's OHLC for Camarilla pivot calculation (using completed prior 1d bar)
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Extract prior bar's OHLC (1d timeframe) - shift by 1 to use only completed prior bar
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close = np.roll(df_1d['close'].values, 1)
    # Set first value to NaN as we don't have prior bar
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Align prior bar OHLC to 6h timeframe
    prior_high_aligned = align_htf_to_ltf(prices, df_1d, prior_high)
    prior_low_aligned = align_htf_to_ltf(prices, df_1d, prior_low)
    prior_close_aligned = align_htf_to_ltf(prices, df_1d, prior_close)
    
    # Calculate Camarilla levels for each 6h bar based on prior bar's OHLC
    # Camarilla R3 = Close + (High - Low) * 1.1/4
    # Camarilla S3 = Close - (High - Low) * 1.1/4
    range_hl = prior_high_aligned - prior_low_aligned
    r3 = prior_close_aligned + range_hl * 1.1 / 4
    s3 = prior_close_aligned - range_hl * 1.1 / 4
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 1) + 1  # EMA50 warmup + 1 for prior bar shift
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_1d_aligned[i]
        
        # Camarilla levels
        r3_level = r3[i]
        s3_level = s3[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below S3 (mean reversion to median)
            if curr_close < s3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (mean reversion to median)
            if curr_close > r3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above R3 AND price > 1d EMA50 AND volume confirmation
            if curr_close > r3_level and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND price < 1d EMA50 AND volume confirmation
            elif curr_close < s3_level and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation
# Long when price breaks above R3 AND price > 1d EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below S3 AND price < 1d EMA50 AND volume > 2.0x 20-bar avg
# Exit when price crosses opposite Camarilla level (S3 for longs, R3 for shorts)
# Uses discrete position sizing (0.25) to minimize fee churn while capturing moves.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h.
# Camarilla R3/S3 are strong intraday levels with good breakout reliability.
# 1d EMA50 filters counter-trend moves, volume spike ensures institutional participation.
# Works in bull markets (trend continuation via breakouts) and bear markets (mean reversion within trend via exits).

name = "6h_Camarilla_R3S3_Breakout_1dEMA50_VolumeSpike_v1"
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
    
    # Get 1d data for EMA50 trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get prior bar's OHLC for Camarilla pivot calculation (using completed prior 1d bar)
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Extract prior bar's OHLC (1d timeframe) - shift by 1 to use only completed prior bar
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close = np.roll(df_1d['close'].values, 1)
    # Set first value to NaN as we don't have prior bar
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Align prior bar OHLC to 6h timeframe
    prior_high_aligned = align_htf_to_ltf(prices, df_1d, prior_high)
    prior_low_aligned = align_htf_to_ltf(prices, df_1d, prior_low)
    prior_close_aligned = align_htf_to_ltf(prices, df_1d, prior_close)
    
    # Calculate Camarilla levels for each 6h bar based on prior bar's OHLC
    # Camarilla R3 = Close + (High - Low) * 1.1/4
    # Camarilla S3 = Close - (High - Low) * 1.1/4
    range_hl = prior_high_aligned - prior_low_aligned
    r3 = prior_close_aligned + range_hl * 1.1 / 4
    s3 = prior_close_aligned - range_hl * 1.1 / 4
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 1) + 1  # EMA50 warmup + 1 for prior bar shift
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_1d_aligned[i]
        
        # Camarilla levels
        r3_level = r3[i]
        s3_level = s3[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below S3 (mean reversion to median)
            if curr_close < s3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (mean reversion to median)
            if curr_close > r3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above R3 AND price > 1d EMA50 AND volume confirmation
            if curr_close > r3_level and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND price < 1d EMA50 AND volume confirmation
            elif curr_close < s3_level and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation
# Long when price breaks above R3 AND price > 1d EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below S3 AND price < 1d EMA50 AND volume > 2.0x 20-bar avg
# Exit when price crosses opposite Camarilla level (S3 for longs, R3 for shorts)
# Uses discrete position sizing (0.25) to minimize fee churn while capturing moves.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h.
# Camarilla R3/S3 are strong intraday levels with good breakout reliability.
# 1d EMA50 filters counter-trend moves, volume spike ensures institutional participation.
# Works in bull markets (trend continuation via breakouts) and bear markets (mean reversion within trend via exits).

name = "6h_Camarilla_R3S3_Breakout_1dEMA50_VolumeSpike_v1"
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
    
    # Get 1d data for EMA50 trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get prior bar's OHLC for Camarilla pivot calculation (using completed prior 1d bar)
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Extract prior bar's OHLC (1d timeframe) - shift by 1 to use only completed prior bar
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_close = np.roll(df_1d['close'].values, 1)
    # Set first value to NaN as we don't have prior bar
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Align prior bar OHLC to 6h timeframe
    prior_high_aligned = align_htf_to_ltf(prices, df_1d, prior_high)
    prior_low_aligned = align_htf_to_ltf(prices, df_1d, prior_low)
    prior_close_aligned = align_htf_to_ltf(prices, df_1d, prior_close)
    
    # Calculate Camarilla levels for each 6h bar based on prior bar's OHLC
    # Camarilla R3 = Close + (High - Low) * 1.1/4
    # Camarilla S3 = Close - (High - Low) * 1.1/4
    range_hl = prior_high_aligned - prior_low_aligned
    r3 = prior_close_aligned + range_hl * 1.1 / 4
    s3 = prior_close_aligned - range_hl * 1.1 / 4
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 1) + 1  # EMA50 warmup + 1 for prior bar shift
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_1d_aligned[i]
        
        # Camarilla levels
        r3_level = r3[i]
        s3_level = s3[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below S3 (mean reversion to median)
            if curr_close < s3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above R3 (mean reversion to median)
            if curr_close > r3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above R3 AND price > 1d EMA50 AND volume confirmation
            if curr_close > r3_level and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 AND price < 1d EMA50 AND volume confirmation
            elif curr_close < s3_level and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation
# Long when price breaks above R3 AND price > 1d EMA50 AND volume > 2.0x 20-bar avg
# Short when price breaks below S3 AND price < 1d EMA50 AND volume > 2.0x 20-bar avg
# Exit when price crosses opposite Camarilla level (S3 for longs, R3 for shorts)
# Uses discrete position sizing (0.25) to minimize fee churn while capturing moves.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h.
# Camarilla R3/S3 are strong intraday levels with good breakout reliability.
# 1d EMA50 filters counter-trend moves, volume spike ensures institutional participation.
# Works in bull markets (trend continuation via breakouts) and bear markets (mean reversion within trend via exits).

name = "6h_Camarilla_R3S3_Breakout_1dEMA50_VolumeSpike_v1"
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
    
    # Get 1d data for EMA50 trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d