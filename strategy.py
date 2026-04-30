#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted Average Price (VWAP) deviation strategy with 12h trend filter.
# Enters long when price deviates significantly below VWAP (mean reversion) in a 12h uptrend,
# and short when price deviates significantly above VWAP in a 12h downtrend.
# Uses volume confirmation to filter low-liquidity false signals.
# Designed to work in both bull and bear markets by aligning with higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete position sizing at ±0.25.

name = "6h_VWAP_Deviation_12hTrend_VolumeFilter_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 6h data ONCE before loop for VWAP calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:  # Need at least 20 periods for VWAP
        return np.zeros(n)
    
    # Calculate typical price and VWAP components
    typical_price = (df_6h['high'] + df_6h['low'] + df_6h['close']) / 3
    vol_times_typical = typical_price * df_6h['volume']
    
    # Cumulative VWAP (reset daily) - using 6h bars, so reset every 4 bars (24h/6h)
    cum_vol = np.cumsum(df_6h['volume'].values)
    cum_vol_times_tp = np.cumsum(vol_times_typical.values)
    vwap_raw = cum_vol_times_tp / cum_vol
    
    # To avoid look-ahead, we use previous bar's VWAP (standard practice)
    vwap = np.concatenate([[vwap_raw[0]], vwap_raw[:-1]])
    
    # Align VWAP to primary timeframe (6h -> 6h: identity but using helper for consistency)
    vwap_aligned = align_htf_to_ltf(prices, df_6h, vwap)
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Standard deviation of price-VWAP for volatility normalization (20-period)
    price_vwap_diff = close - vwap_aligned
    price_vwap_diff_abs = np.abs(price_vwap_diff)
    stdev_20 = pd.Series(price_vwap_diff_abs).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50) + 1  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(vwap_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(stdev_20[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vwap = vwap_aligned[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_stdev = stdev_20[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Calculate normalized deviation from VWAP
        if curr_stdev > 0:
            dev_norm = (curr_close - curr_vwap) / curr_stdev
        else:
            dev_norm = 0
        
        if position == 0:  # Flat - look for new entries
            # Long: price significantly below VWAP (-2.0 deviation), 12h uptrend, volume confirmation
            if (dev_norm < -2.0 and 
                curr_close > curr_ema_50_12h and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price significantly above VWAP (+2.0 deviation), 12h downtrend, volume confirmation
            elif (dev_norm > 2.0 and 
                  curr_close < curr_ema_50_12h and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit when price returns to VWAP (mean reversion achieved)
            if curr_close >= curr_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price returns to VWAP (mean reversion achieved)
            if curr_close <= curr_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted Average Price (VWAP) deviation strategy with 12h trend filter.
# Enters long when price deviates significantly below VWAP (mean reversion) in a 12h uptrend,
# and short when price deviates significantly above VWAP in a 12h downtrend.
# Uses volume confirmation to filter low-liquidity false signals.
# Designed to work in both bull and bear markets by aligning with higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete position sizing at ±0.25.

name = "6h_VWAP_Deviation_12hTrend_VolumeFilter_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 6h data ONCE before loop for VWAP calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:  # Need at least 20 periods for VWAP
        return np.zeros(n)
    
    # Calculate typical price and VWAP components
    typical_price = (df_6h['high'] + df_6h['low'] + df_6h['close']) / 3
    vol_times_typical = typical_price * df_6h['volume']
    
    # Cumulative VWAP (reset daily) - using 6h bars, so reset every 4 bars (24h/6h)
    cum_vol = np.cumsum(df_6h['volume'].values)
    cum_vol_times_tp = np.cumsum(vol_times_typical.values)
    vwap_raw = cum_vol_times_tp / cum_vol
    
    # To avoid look-ahead, we use previous bar's VWAP (standard practice)
    vwap = np.concatenate([[vwap_raw[0]], vwap_raw[:-1]])
    
    # Align VWAP to primary timeframe (6h -> 6h: identity but using helper for consistency)
    vwap_aligned = align_htf_to_ltf(prices, df_6h, vwap)
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Standard deviation of price-VWAP for volatility normalization (20-period)
    price_vwap_diff = close - vwap_aligned
    price_vwap_diff_abs = np.abs(price_vwap_diff)
    stdev_20 = pd.Series(price_vwap_diff_abs).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50) + 1  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(vwap_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(stdev_20[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vwap = vwap_aligned[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_stdev = stdev_20[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Calculate normalized deviation from VWAP
        if curr_stdev > 0:
            dev_norm = (curr_close - curr_vwap) / curr_stdev
        else:
            dev_norm = 0
        
        if position == 0:  # Flat - look for new entries
            # Long: price significantly below VWAP (-2.0 deviation), 12h uptrend, volume confirmation
            if (dev_norm < -2.0 and 
                curr_close > curr_ema_50_12h and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price significantly above VWAP (+2.0 deviation), 12h downtrend, volume confirmation
            elif (dev_norm > 2.0 and 
                  curr_close < curr_ema_50_12h and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit when price returns to VWAP (mean reversion achieved)
            if curr_close >= curr_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price returns to VWAP (mean reversion achieved)
            if curr_close <= curr_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted Average Price (VWAP) deviation strategy with 12h trend filter.
# Enters long when price deviates significantly below VWAP (mean reversion) in a 12h uptrend,
# and short when price deviates significantly above VWAP in a 12h downtrend.
# Uses volume confirmation to filter low-liquidity false signals.
# Designed to work in both bull and bear markets by aligning with higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete position sizing at ±0.25.

name = "6h_VWAP_Deviation_12hTrend_VolumeFilter_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 6h data ONCE before loop for VWAP calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:  # Need at least 20 periods for VWAP
        return np.zeros(n)
    
    # Calculate typical price and VWAP components
    typical_price = (df_6h['high'] + df_6h['low'] + df_6h['close']) / 3
    vol_times_typical = typical_price * df_6h['volume']
    
    # Cumulative VWAP (reset daily) - using 6h bars, so reset every 4 bars (24h/6h)
    cum_vol = np.cumsum(df_6h['volume'].values)
    cum_vol_times_tp = np.cumsum(vol_times_typical.values)
    vwap_raw = cum_vol_times_tp / cum_vol
    
    # To avoid look-ahead, we use previous bar's VWAP (standard practice)
    vwap = np.concatenate([[vwap_raw[0]], vwap_raw[:-1]])
    
    # Align VWAP to primary timeframe (6h -> 6h: identity but using helper for consistency)
    vwap_aligned = align_htf_to_ltf(prices, df_6h, vwap)
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Standard deviation of price-VWAP for volatility normalization (20-period)
    price_vwap_diff = close - vwap_aligned
    price_vwap_diff_abs = np.abs(price_vwap_diff)
    stdev_20 = pd.Series(price_vwap_diff_abs).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50) + 1  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(vwap_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(stdev_20[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vwap = vwap_aligned[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_stdev = stdev_20[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Calculate normalized deviation from VWAP
        if curr_stdev > 0:
            dev_norm = (curr_close - curr_vwap) / curr_stdev
        else:
            dev_norm = 0
        
        if position == 0:  # Flat - look for new entries
            # Long: price significantly below VWAP (-2.0 deviation), 12h uptrend, volume confirmation
            if (dev_norm < -2.0 and 
                curr_close > curr_ema_50_12h and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price significantly above VWAP (+2.0 deviation), 12h downtrend, volume confirmation
            elif (dev_norm > 2.0 and 
                  curr_close < curr_ema_50_12h and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit when price returns to VWAP (mean reversion achieved)
            if curr_close >= curr_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price returns to VWAP (mean reversion achieved)
            if curr_close <= curr_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted Average Price (VWAP) deviation strategy with 12h trend filter.
# Enters long when price deviates significantly below VWAP (mean reversion) in a 12h uptrend,
# and short when price deviates significantly above VWAP in a 12h downtrend.
# Uses volume confirmation to filter low-liquidity false signals.
# Designed to work in both bull and bear markets by aligning with higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete position sizing at ±0.25.

name = "6h_VWAP_Deviation_12hTrend_VolumeFilter_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 6h data ONCE before loop for VWAP calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:  # Need at least 20 periods for VWAP
        return np.zeros(n)
    
    # Calculate typical price and VWAP components
    typical_price = (df_6h['high'] + df_6h['low'] + df_6h['close']) / 3
    vol_times_typical = typical_price * df_6h['volume']
    
    # Cumulative VWAP (reset daily) - using 6h bars, so reset every 4 bars (24h/6h)
    cum_vol = np.cumsum(df_6h['volume'].values)
    cum_vol_times_tp = np.cumsum(vol_times_typical.values)
    vwap_raw = cum_vol_times_tp / cum_vol
    
    # To avoid look-ahead, we use previous bar's VWAP (standard practice)
    vwap = np.concatenate([[vwap_raw[0]], vwap_raw[:-1]])
    
    # Align VWAP to primary timeframe (6h -> 6h: identity but using helper for consistency)
    vwap_aligned = align_htf_to_ltf(prices, df_6h, vwap)
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Standard deviation of price-VWAP for volatility normalization (20-period)
    price_vwap_diff = close - vwap_aligned
    price_vwap_diff_abs = np.abs(price_vwap_diff)
    stdev_20 = pd.Series(price_vwap_diff_abs).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50) + 1  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(vwap_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(stdev_20[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vwap = vwap_aligned[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_stdev = stdev_20[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Calculate normalized deviation from VWAP
        if curr_stdev > 0:
            dev_norm = (curr_close - curr_vwap) / curr_stdev
        else:
            dev_norm = 0
        
        if position == 0:  # Flat - look for new entries
            # Long: price significantly below VWAP (-2.0 deviation), 12h uptrend, volume confirmation
            if (dev_norm < -2.0 and 
                curr_close > curr_ema_50_12h and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price significantly above VWAP (+2.0 deviation), 12h downtrend, volume confirmation
            elif (dev_norm > 2.0 and 
                  curr_close < curr_ema_50_12h and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit when price returns to VWAP (mean reversion achieved)
            if curr_close >= curr_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price returns to VWAP (mean reversion achieved)
            if curr_close <= curr_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted Average Price (VWAP) deviation strategy with 12h trend filter.
# Enters long when price deviates significantly below VWAP (mean reversion) in a 12h uptrend,
# and short when price deviates significantly above VWAP in a 12h downtrend.
# Uses volume confirmation to filter low-liquidity false signals.
# Designed to work in both bull and bear markets by aligning with higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete position sizing at ±0.25.

name = "6h_VWAP_Deviation_12hTrend_VolumeFilter_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 6h data ONCE before loop for VWAP calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:  # Need at least 20 periods for VWAP
        return np.zeros(n)
    
    # Calculate typical price and VWAP components
    typical_price = (df_6h['high'] + df_6h['low'] + df_6h['close']) / 3
    vol_times_typical = typical_price * df_6h['volume']
    
    # Cumulative VWAP (reset daily) - using 6h bars, so reset every 4 bars (24h/6h)
    cum_vol = np.cumsum(df_6h['volume'].values)
    cum_vol_times_tp = np.cumsum(vol_times_typical.values)
    vwap_raw = cum_vol_times_tp / cum_vol
    
    # To avoid look-ahead, we use previous bar's VWAP (standard practice)
    vwap = np.concatenate([[vwap_raw[0]], vwap_raw[:-1]])
    
    # Align VWAP to primary timeframe (6h -> 6h: identity but using helper for consistency)
    vwap_aligned = align_htf_to_ltf(prices, df_6h, vwap)
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Standard deviation of price-VWAP for volatility normalization (20-period)
    price_vwap_diff = close - vwap_aligned
    price_vwap_diff_abs = np.abs(price_vwap_diff)
    stdev_20 = pd.Series(price_vwap_diff_abs).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50) + 1  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(vwap_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(stdev_20[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vwap = vwap_aligned[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_stdev = stdev_20[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Calculate normalized deviation from VWAP
        if curr_stdev > 0:
            dev_norm = (curr_close - curr_vwap) / curr_stdev
        else:
            dev_norm = 0
        
        if position == 0:  # Flat - look for new entries
            # Long: price significantly below VWAP (-2.0 deviation), 12h uptrend, volume confirmation
            if (dev_norm < -2.0 and 
                curr_close > curr_ema_50_12h and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price significantly above VWAP (+2.0 deviation), 12h downtrend, volume confirmation
            elif (dev_norm > 2.0 and 
                  curr_close < curr_ema_50_12h and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit when price returns to VWAP (mean reversion achieved)
            if curr_close >= curr_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price returns to VWAP (mean reversion achieved)
            if curr_close <= curr_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted Average Price (VWAP) deviation strategy with 12h trend filter.
# Enters long when price deviates significantly below VWAP (mean reversion) in a 12h uptrend,
# and short when price deviates significantly above VWAP in a 12h downtrend.
# Uses volume confirmation to filter low-liquidity false signals.
# Designed to work in both bull and bear markets by aligning with higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete position sizing at ±0.25.

name = "6h_VWAP_Deviation_12hTrend_VolumeFilter_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 6h data ONCE before loop for VWAP calculation
    df_