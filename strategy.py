#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 12-hour RSI filter and volume confirmation.
# Long when price breaks above 4h Donchian upper (20) and 12h RSI > 50.
# Short when price breaks below 4h Donchian lower (20) and 12h RSI < 50.
# Uses volume > 1.5x 20-period average to confirm breakout strength.
# Target: 75-200 total trades over 4 years (19-50/year) to stay within optimal range.

name = "4h_donchian20_12h_rsi_vol_v1"
timeframe = "4h"
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
    
    # 4h Donchian Channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # 12h RSI (14-period)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    delta = close_12h_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_12h = (100 - (100 / (1 + rs))).values
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if RSI data not available
        if np.isnan(rsi_12h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or Donchian middle reversion
        if position == 1:  # long position
            # Exit: price breaks below Donchian lower or reverse signal
            if (close[i] < donchian_lower[i] or 
                (close[i] > donchian_upper[i] and rsi_12h_aligned[i] < 50)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian upper or reverse signal
            if (close[i] > donchian_upper[i] or 
                (close[i] < donchian_lower[i] and rsi_12h_aligned[i] > 50)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: break above upper with RSI > 50 and volume confirmation
            if (close[i] > donchian_upper[i] and 
                rsi_12h_aligned[i] > 50 and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower with RSI < 50 and volume confirmation
            elif (close[i] < donchian_lower[i] and 
                  rsi_12h_aligned[i] < 50 and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 12-hour RSI filter and volume confirmation.
# Long when price breaks above 4h Donchian upper (20) and 12h RSI > 50.
# Short when price breaks below 4h Donchian lower (20) and 12h RSI < 50.
# Uses volume > 1.5x 20-period average to confirm breakout strength.
# Target: 75-200 total trades over 4 years (19-50/year) to stay within optimal range.

name = "4h_donchian20_12h_rsi_vol_v1"
timeframe = "4h"
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
    
    # 4h Donchian Channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # 12h RSI (14-period)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    delta = close_12h_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_12h = (100 - (100 / (1 + rs))).values
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if RSI data not available
        if np.isnan(rsi_12h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or Donchian middle reversion
        if position == 1:  # long position
            # Exit: price breaks below Donchian lower or reverse signal
            if (close[i] < donchian_lower[i] or 
                (close[i] > donchian_upper[i] and rsi_12h_aligned[i] < 50)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian upper or reverse signal
            if (close[i] > donchian_upper[i] or 
                (close[i] < donchian_lower[i] and rsi_12h_aligned[i] > 50)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: break above upper with RSI > 50 and volume confirmation
            if (close[i] > donchian_upper[i] and 
                rsi_12h_aligned[i] > 50 and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower with RSI < 50 and volume confirmation
            elif (close[i] < donchian_lower[i] and 
                  rsi_12h_aligned[i] < 50 and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals