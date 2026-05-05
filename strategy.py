#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h volume spike and 1d ADX trend filter
# Long when: price breaks above R3, volume > 2x 20-period average on 12h, and 1d ADX > 25
# Short when: price breaks below S3, volume > 2x 20-period average on 12h, and 1d ADX > 25
# Exit when price returns to Camarilla R3/S3 level (mean reversion) or opposite breakout
# Uses Camarilla levels from 1d for structure, 12h for volume confirmation, and 1d ADX for trend strength.
# Works in both bull (breakout continuation with strong trend) and bear (mean reversion via exits) markets.
# Timeframe: 6h, HTF: 12h/1d. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Camarilla_R3S3_Breakout_12hVolume_1dADX25"
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
    open_price = prices['open'].values
    
    # Get 12h data ONCE before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    if len(volume_12h) >= 20:
        vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
        volume_filter_12h = volume_12h > (2.0 * vol_ma_20_12h)
    else:
        volume_filter_12h = np.zeros(len(df_12h), dtype=bool)
    
    # Align 12h volume filter to 6h timeframe
    volume_filter_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_filter_12h)
    
    # Get 1d data ONCE before loop for Camarilla levels and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX trend filter (ADX > 25 indicates strong trend)
    if len(high_1d) >= 14:
        # Calculate True Range
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First value has no TR
        
        # Calculate Directional Movement
        up_move = np.diff(high_1d)
        down_move = -np.diff(low_1d)
        up_move = np.concatenate([[0], up_move])
        down_move = np.concatenate([[0], down_move])
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
                else:
                    result[i] = np.nan
            return result
        
        atr = wilder_smooth(tr, 14)
        plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
        minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smooth(dx, 14)
        adx_filter = adx > 25
    else:
        adx_filter = np.zeros(len(close_1d), dtype=bool)
    
    # Align 1d ADX filter to 6h timeframe
    adx_filter_aligned = align_htf_to_ltf(prices, df_1d, adx_filter)
    
    # Calculate Camarilla levels from previous 1d bar
    if len(high_1d) >= 2:
        prev_high = np.roll(high_1d, 1)
        prev_low = np.roll(low_1d, 1)
        prev_close = np.roll(close_1d, 1)
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        rang = prev_high - prev_low
        camarilla_r3 = prev_close + 1.1 * rang * 1.1 / 4
        camarilla_s3 = prev_close - 1.1 * rang * 1.1 / 4
    else:
        camarilla_r3 = np.full(len(close_1d), np.nan)
        camarilla_s3 = np.full(len(close_1d), np.nan)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) if 'ema_50_1d_aligned' in locals() else False) or \
           np.isnan(camarilla_r3_aligned[i]) or \
           np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(volume_filter_12h_aligned[i]) or \
           np.isnan(adx_filter_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3, volume filter, and ADX > 25
            if (close[i] > camarilla_r3_aligned[i] and 
                open_price[i] <= camarilla_r3_aligned[i] and  # Ensure breakout happens on this bar
                volume_filter_12h_aligned[i] and 
                adx_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3, volume filter, and ADX > 25
            elif (close[i] < camarilla_s3_aligned[i] and 
                  open_price[i] >= camarilla_s3_aligned[i] and  # Ensure breakdown happens on this bar
                  volume_filter_12h_aligned[i] and 
                  adx_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below R3 (mean reversion) or breaks below S3 (reversal)
            if close[i] < camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above S3 (mean reversion) or breaks above R3 (reversal)
            if close[i] > camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h volume spike and 1d ADX trend filter
# Long when: price breaks above R3, volume > 2x 20-period average on 12h, and 1d ADX > 25
# Short when: price breaks below S3, volume > 2x 20-period average on 12h, and 1d ADX > 25
# Exit when price returns to Camarilla R3/S3 level (mean reversion) or opposite breakout
# Uses Camarilla levels from 1d for structure, 12h for volume confirmation, and 1d ADX for trend strength.
# Works in both bull (breakout continuation with strong trend) and bear (mean reversion via exits) markets.
# Timeframe: 6h, HTF: 12h/1d. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Camarilla_R3S3_Breakout_12hVolume_1dADX25"
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
    open_price = prices['open'].values
    
    # Get 12h data ONCE before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    if len(volume_12h) >= 20:
        vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
        volume_filter_12h = volume_12h > (2.0 * vol_ma_20_12h)
    else:
        volume_filter_12h = np.zeros(len(df_12h), dtype=bool)
    
    # Align 12h volume filter to 6h timeframe
    volume_filter_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_filter_12h)
    
    # Get 1d data ONCE before loop for Camarilla levels and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX trend filter (ADX > 25 indicates strong trend)
    if len(high_1d) >= 14:
        # Calculate True Range
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First value has no TR
        
        # Calculate Directional Movement
        up_move = np.diff(high_1d)
        down_move = -np.diff(low_1d)
        up_move = np.concatenate([[0], up_move])
        down_move = np.concatenate([[0], down_move])
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
                else:
                    result[i] = np.nan
            return result
        
        atr = wilder_smooth(tr, 14)
        plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
        minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smooth(dx, 14)
        adx_filter = adx > 25
    else:
        adx_filter = np.zeros(len(close_1d), dtype=bool)
    
    # Align 1d ADX filter to 6h timeframe
    adx_filter_aligned = align_htf_to_ltf(prices, df_1d, adx_filter)
    
    # Calculate Camarilla levels from previous 1d bar
    if len(high_1d) >= 2:
        prev_high = np.roll(high_1d, 1)
        prev_low = np.roll(low_1d, 1)
        prev_close = np.roll(close_1d, 1)
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        rang = prev_high - prev_low
        camarilla_r3 = prev_close + 1.1 * rang * 1.1 / 4
        camarilla_s3 = prev_close - 1.1 * rang * 1.1 / 4
    else:
        camarilla_r3 = np.full(len(close_1d), np.nan)
        camarilla_s3 = np.full(len(close_1d), np.nan)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if np.isnan(camarilla_r3_aligned[i]) or \
           np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(volume_filter_12h_aligned[i]) or \
           np.isnan(adx_filter_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3, volume filter, and ADX > 25
            if (close[i] > camarilla_r3_aligned[i] and 
                open_price[i] <= camarilla_r3_aligned[i] and  # Ensure breakout happens on this bar
                volume_filter_12h_aligned[i] and 
                adx_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3, volume filter, and ADX > 25
            elif (close[i] < camarilla_s3_aligned[i] and 
                  open_price[i] >= camarilla_s3_aligned[i] and  # Ensure breakdown happens on this bar
                  volume_filter_12h_aligned[i] and 
                  adx_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below R3 (mean reversion) or breaks below S3 (reversal)
            if close[i] < camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above S3 (mean reversion) or breaks above R3 (reversal)
            if close[i] > camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals