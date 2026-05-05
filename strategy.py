#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3 AND price > 1d EMA34 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below Camarilla S3 AND price < 1d EMA34 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses Camarilla H3/L3 midpoint OR volume < avg_volume(20)
# Uses discrete sizing 0.25 to minimize fee churn
# Target: 75-200 total trades over 4 years (19-50/year)
# Camarilla levels from 1d provide intraday support/resistance; 1d EMA34 filters primary trend; volume spike confirms breakout strength
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for calculations
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: H4 = Close + 1.5*(High-Low), L4 = Close - 1.5*(High-Low)
    # R3 = Close + 1.1*(High-Low), S3 = Close - 1.1*(High-Low)
    # H3 = Close + 1.0*(High-Low), L3 = Close - 1.0*(High-Low)
    # We use R3/S3 for entry, H3/L3 for exit
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First value will be incorrect due to roll, but we'll handle with min_periods
    prev_close[0] = close_1d[0]  # fallback
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    camarilla_range = prev_high - prev_low
    camarilla_r3 = prev_close + 1.1 * camarilla_range
    camarilla_s3 = prev_close - 1.1 * camarilla_range
    camarilla_h3 = prev_close + 1.0 * camarilla_range
    camarilla_l3 = prev_close - 1.0 * camarilla_range
    camarilla_mid = (camarilla_h3 + camarilla_l3) / 2  # H3/L3 midpoint for exit
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R3, above 1d EMA34, volume confirmation
            if close[i] > camarilla_r3[i] and close[i] > ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3, below 1d EMA34, volume confirmation
            elif close[i] < camarilla_s3[i] and close[i] < ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Camarilla H3/L3 midpoint OR volume drops below average
            if close[i] < camarilla_mid[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above Camarilla H3/L3 midpoint OR volume drops below average
            if close[i] > camarilla_mid[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3 AND price > 1d EMA34 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below Camarilla S3 AND price < 1d EMA34 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses Camarilla H3/L3 midpoint OR volume < avg_volume(20)
# Uses discrete sizing 0.25 to minimize fee churn
# Target: 75-200 total trades over 4 years (19-50/year)
# Camarilla levels from 1d provide intraday support/resistance; 1d EMA34 filters primary trend; volume spike confirms breakout strength
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for calculations
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: H4 = Close + 1.5*(High-Low), L4 = Close - 1.5*(High-Low)
    # R3 = Close + 1.1*(High-Low), S3 = Close - 1.1*(High-Low)
    # H3 = Close + 1.0*(High-Low), L3 = Close - 1.0*(High-Low)
    # We use R3/S3 for entry, H3/L3 for exit
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First value will be incorrect due to roll, but we'll handle with min_periods
    prev_close[0] = close_1d[0]  # fallback
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    camarilla_range = prev_high - prev_low
    camarilla_r3 = prev_close + 1.1 * camarilla_range
    camarilla_s3 = prev_close - 1.1 * camarilla_range
    camarilla_h3 = prev_close + 1.0 * camarilla_range
    camarilla_l3 = prev_close - 1.0 * camarilla_range
    camarilla_mid = (camarilla_h3 + camarilla_l3) / 2  # H3/L3 midpoint for exit
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R3, above 1d EMA34, volume confirmation
            if close[i] > camarilla_r3[i] and close[i] > ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3, below 1d EMA34, volume confirmation
            elif close[i] < camarilla_s3[i] and close[i] < ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Camarilla H3/L3 midpoint OR volume drops below average
            if close[i] < camarilla_mid[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above Camarilla H3/L3 midpoint OR volume drops below average
            if close[i] > camarilla_mid[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals