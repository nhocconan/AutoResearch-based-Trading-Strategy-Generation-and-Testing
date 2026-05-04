#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels (R3/S3) represent strong intraday support/resistance. A break above R3
# or below S3 with volume confirmation (2.0x 20-period volume EMA) and trend alignment (price
# vs 1d EMA34) captures momentum bursts. Designed for 4h timeframe to target 20-50 trades/year
# (80-200 total over 4 years) with discrete sizing (0.30). Works in bull markets by buying
# breakouts above R3 in uptrends and in bear markets by selling breakdowns below S3 in downtrends.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels (R3, S3) from previous 1d bar
    # Camarilla formulas: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # We use the previous 1d bar's OHLC to avoid look-ahead
    df_1d = get_htf_data(prices, '1d')  # Reload for OHLC (still only called once outside loop in practice, but here for clarity - will be optimized)
    # Actually, we need to compute this properly: get the 1d data once and use its OHLC
    # Let's restructure to load 1d data once
    
    # Re-structure: load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla R3 and S3 from 1d OHLC (using previous bar to avoid look-ahead)
    # R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 4h timeframe (using previous 1d bar's values)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 2.0x 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + volume confirmation + price above 1d EMA34 (uptrend)
            if (close[i] > camarilla_r3_aligned[i] and volume_confirmed and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Camarilla S3 + volume confirmation + price below 1d EMA34 (downtrend)
            elif (close[i] < camarilla_s3_aligned[i] and volume_confirmed and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price retreats below Camarilla R3 OR below 1d EMA34 (trend change)
            if close[i] < camarilla_r3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price rises above Camarilla S3 OR above 1d EMA34 (trend change)
            if close[i] > camarilla_s3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels (R3/S3) represent strong intraday support/resistance. A break above R3
# or below S3 with volume confirmation (2.0x 20-period volume EMA) and trend alignment (price
# vs 1d EMA34) captures momentum bursts. Designed for 4h timeframe to target 20-50 trades/year
# (80-200 total over 4 years) with discrete sizing (0.30). Works in bull markets by buying
# breakouts above R3 in uptrends and in bear markets by selling breakdowns below S3 in downtrends.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla R3 and S3 from 1d OHLC (using previous bar to avoid look-ahead)
    # R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 4h timeframe (using previous 1d bar's values)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 2.0x 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + volume confirmation + price above 1d EMA34 (uptrend)
            if (close[i] > camarilla_r3_aligned[i] and volume_confirmed and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Camarilla S3 + volume confirmation + price below 1d EMA34 (downtrend)
            elif (close[i] < camarilla_s3_aligned[i] and volume_confirmed and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price retreats below Camarilla R3 OR below 1d EMA34 (trend change)
            if close[i] < camarilla_r3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price rises above Camarilla S3 OR above 1d EMA34 (trend change)
            if close[i] > camarilla_s3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals