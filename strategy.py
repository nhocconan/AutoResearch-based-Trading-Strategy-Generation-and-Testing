#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1-week trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) during bullish weekly trend with volume > 1.5x 20-period average.
# Short when Williams %R > -20 (overbought) during bearish weekly trend with volume confirmation.
# Uses weekly trend to avoid counter-trend trades. Williams %R provides mean reversion signals in ranging markets.
# Target: 100-200 total trades over 4 years (25-50/year) to stay within optimal range.

name = "6h_williamsr_1w_trend_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    highest_high = high_series.rolling(window=14, min_periods=14).max()
    lowest_low = low_series.rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close_series) / (highest_high - lowest_low)
    williams_r = williams_r.replace(0, np.nan).fillna(0).values  # Handle division by zero
    
    # Weekly trend filter: bullish/bearish week based on close vs open
    df_1w = get_htf_data(prices, '1w')
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open  # True for bullish week
    weekly_bearish = weekly_close < weekly_open   # True for bearish week
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if weekly trend data not available
        if np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: Williams %R > -50 (overbought) or weekly turn bearish
            if (williams_r[i] > -50 or 
                weekly_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R < -50 (oversold) or weekly turn bullish
            if (williams_r[i] < -50 or 
                weekly_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and weekly trend filter
            if volume_filter:
                # Long: Williams %R < -80 (oversold) during bullish week
                if (williams_r[i] < -80 and 
                    weekly_bullish_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R > -20 (overbought) during bearish week
                elif (williams_r[i] > -20 and 
                      weekly_bearish_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1-week trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) during bullish weekly trend with volume > 1.5x 20-period average.
# Short when Williams %R > -20 (overbought) during bearish weekly trend with volume confirmation.
# Uses weekly trend to avoid counter-trend trades. Williams %R provides mean reversion signals in ranging markets.
# Target: 100-200 total trades over 4 years (25-50/year) to stay within optimal range.

name = "6h_williamsr_1w_trend_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    highest_high = high_series.rolling(window=14, min_periods=14).max()
    lowest_low = low_series.rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close_series) / (highest_high - lowest_low)
    williams_r = williams_r.replace(0, np.nan).fillna(0).values  # Handle division by zero
    
    # Weekly trend filter: bullish/bearish week based on close vs open
    df_1w = get_htf_data(prices, '1w')
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open  # True for bullish week
    weekly_bearish = weekly_close < weekly_open   # True for bearish week
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if weekly trend data not available
        if np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: Williams %R > -50 (overbought) or weekly turn bearish
            if (williams_r[i] > -50 or 
                weekly_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R < -50 (oversold) or weekly turn bullish
            if (williams_r[i] < -50 or 
                weekly_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and weekly trend filter
            if volume_filter:
                # Long: Williams %R < -80 (oversold) during bullish week
                if (williams_r[i] < -80 and 
                    weekly_bullish_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R > -20 (overbought) during bearish week
                elif (williams_r[i] > -20 and 
                      weekly_bearish_aligned[i]):
                    signals[i] = -0.25
                    position = -1
    
    return signals