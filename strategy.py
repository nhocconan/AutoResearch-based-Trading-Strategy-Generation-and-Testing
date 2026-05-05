#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA20 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND 4h close > 4h EMA20 AND volume > 1.5x 20 EMA
# Short when price breaks below Camarilla S3 AND 4h close < 4h EMA20 AND volume > 1.5x 20 EMA
# Uses discrete sizing (0.20) to limit fee drag. Target: 15-30 trades/year per symbol.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# Uses 4h for HTF trend to avoid counter-trend trades and 1h for Camarilla timing.

name = "1h_Camarilla_R3S3_4hEMA20_VolumeConfirm"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1h data for Camarilla calculation (using 1h OHLC)
    # Calculate typical price for Camarilla levels
    typical_price = (high + low + close) / 3.0
    
    # Get 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Uptrend when close > EMA20, downtrend when close < EMA20
    uptrend_4h = close_4h > ema_20_4h
    downtrend_4h = close_4h < ema_20_4h
    
    # Align 4h trend to 1h timeframe
    uptrend_4h_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h.astype(float))
    downtrend_4h_aligned = align_htf_to_ltf(prices, df_4h, downtrend_4h.astype(float))
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    # Calculate Camarilla levels for 1h using typical price
    # Need to calculate Camarilla on completed 1h bars, so shift by 1
    typical_price_shifted = np.roll(typical_price, 1)
    typical_price_shifted[0] = np.nan
    
    # Calculate Camarilla R3 and S3 levels
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    # But using typical price for more stability
    high_low_range = high - low
    camarilla_r3 = typical_price_shifted + 1.1 * high_low_range * 1.1 / 4
    camarilla_s3 = typical_price_shifted - 1.1 * high_low_range * 1.1 / 4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(uptrend_4h_aligned[i]) or np.isnan(downtrend_4h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Camarilla R3 AND 4h uptrend AND volume spike
            if (close[i] > camarilla_r3[i] and 
                uptrend_4h_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price < Camarilla S3 AND 4h downtrend AND volume spike
            elif (close[i] < camarilla_s3[i] and 
                  downtrend_4h_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price < Camarilla S3 OR 4h trend changes to downtrend
            if (close[i] < camarilla_s3[i] or 
                downtrend_4h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price > Camarilla R3 OR 4h trend changes to uptrend
            if (close[i] > camarilla_r3[i] or 
                uptrend_4h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA20 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND 4h close > 4h EMA20 AND volume > 1.5x 20 EMA
# Short when price breaks below Camarilla S3 AND 4h close < 4h EMA20 AND volume > 1.5x 20 EMA
# Uses discrete sizing (0.20) to limit fee drag. Target: 15-30 trades/year per symbol.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# Uses 4h for HTF trend to avoid counter-trend trades and 1h for Camarilla timing.

name = "1h_Camarilla_R3S3_4hEMA20_VolumeConfirm"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1h data for Camarilla calculation (using 1h OHLC)
    # Calculate typical price for Camarilla levels
    typical_price = (high + low + close) / 3.0
    
    # Get 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Uptrend when close > EMA20, downtrend when close < EMA20
    uptrend_4h = close_4h > ema_20_4h
    downtrend_4h = close_4h < ema_20_4h
    
    # Align 4h trend to 1h timeframe
    uptrend_4h_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h.astype(float))
    downtrend_4h_aligned = align_htf_to_ltf(prices, df_4h, downtrend_4h.astype(float))
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)
    
    # Calculate Camarilla levels for 1h using typical price
    # Need to calculate Camarilla on completed 1h bars, so shift by 1
    typical_price_shifted = np.roll(typical_price, 1)
    typical_price_shifted[0] = np.nan
    
    # Calculate Camarilla R3 and S3 levels
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    # But using typical price for more stability
    high_low_range = high - low
    camarilla_r3 = typical_price_shifted + 1.1 * high_low_range * 1.1 / 4
    camarilla_s3 = typical_price_shifted - 1.1 * high_low_range * 1.1 / 4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(uptrend_4h_aligned[i]) or np.isnan(downtrend_4h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Camarilla R3 AND 4h uptrend AND volume spike
            if (close[i] > camarilla_r3[i] and 
                uptrend_4h_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price < Camarilla S3 AND 4h downtrend AND volume spike
            elif (close[i] < camarilla_s3[i] and 
                  downtrend_4h_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price < Camarilla S3 OR 4h trend changes to downtrend
            if (close[i] < camarilla_s3[i] or 
                downtrend_4h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price > Camarilla R3 OR 4h trend changes to uptrend
            if (close[i] > camarilla_r3[i] or 
                uptrend_4h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals