#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d ADX trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In ranging markets (ADX<25), 
# extreme readings (>80 for oversold, <20 for overbought) often precede mean reversion.
# In trending markets (ADX>=25), we fade extremes only when aligned with the trend 
# (long when %R<20 in uptrend, short when %>80 in downtrend) to catch pullbacks.
# Volume confirmation ensures institutional participation. Discrete sizing 0.25.
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years).
# Works in bull markets (trend pullbacks) and bear markets (ranging mean reversion).

name = "6h_WilliamsR_Extreme_1dADX_Volume_v1"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Williams %R(14) - using prior bar to avoid look-ahead
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().shift(1).values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().shift(1).values
    williams_r = np.where(
        (highest_high_14 - lowest_low_14) != 0,
        ((highest_high_14 - close) / (highest_high_14 - lowest_low_14)) * -100,
        0.0  # avoid division by zero
    )
    
    # Calculate 1d ADX(14) trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need enough for ADX calculation
        return np.zeros(n)
    
    # True Range for ADX
    tr_1d = np.maximum(
        df_1d['high'].values - df_1d['low'].values,
        np.maximum(
            np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]])),
            np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))
        )
    )
    # Directional Movement
    dm_plus = np.where(
        (df_1d['high'].values - np.concatenate([[df_1d['high'].values[0]], df_1d['high'].values[:-1]])) > 
        (np.concatenate([[df_1d['low'].values[0]], df_1d['low'].values[:-1]]) - df_1d['low'].values),
        np.maximum(df_1d['high'].values - np.concatenate([[df_1d['high'].values[0]], df_1d['high'].values[:-1]]), 0),
        0
    )
    dm_minus = np.where(
        (np.concatenate([[df_1d['low'].values[0]], df_1d['low'].values[:-1]]) - df_1d['low'].values) > 
        (df_1d['high'].values - np.concatenate([[df_1d['high'].values[0]], df_1d['high'].values[:-1]])),
        np.maximum(np.concatenate([[df_1d['low'].values[0]], df_1d['low'].values[:-1]]) - df_1d['low'].values, 0),
        0
    )
    # Smooth TR, DM+ , DM- with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])  # first value is simple average
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilder_smooth(tr_1d, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    # Avoid division by zero
    di_plus = np.where(atr_1d != 0, (dm_plus_smooth / atr_1d) * 100, 0)
    di_minus = np.where(atr_1d != 0, (dm_minus_smooth / atr_1d) * 100, 0)
    dx = np.where((di_plus + di_minus) != 0, (np.abs(di_plus - di_minus) / (di_plus + di_minus)) * 100, 0)
    adx_14 = wilder_smooth(dx, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Williams %R, ADX, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(adx_14_aligned[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r[i]
        curr_adx = adx_14_aligned[i]
        
        # Trend regime: ADX >= 25 = trending, ADX < 25 = ranging
        trending = curr_adx >= 25.0
        ranging = curr_adx < 25.0
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long conditions:
            # - In ranging market: Williams %R < -80 (oversold) 
            # - In trending market: Williams %R < -20 (pullback) AND price above 20-period EMA (uptrend)
            # - Volume confirmation
            ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean()
            uptrend_filter = curr_close > ema_20.iloc[i] if not np.isnan(ema_20.iloc[i]) else False
            
            long_ranging = ranging and curr_williams_r < -80.0
            long_trending = trending and curr_williams_r < -20.0 and uptrend_filter
            
            # Short conditions:
            # - In ranging market: Williams %R > -20 (overbought)
            # - In trending market: Williams %R > -80 (pullback) AND price below 20-period EMA (downtrend)
            # - Volume confirmation
            downtrend_filter = curr_close < ema_20.iloc[i] if not np.isnan(ema_20.iloc[i]) else False
            
            short_ranging = ranging and curr_williams_r > -20.0
            short_trending = trending and curr_williams_r > -80.0 and downtrend_filter
            
            if (long_ranging or long_trending) and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif (short_ranging or short_trending) and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R > -20 (overbought) OR trend turns down (ADX falling below 20)
            elif curr_williams_r > -20.0 or (trending and curr_adx < 20.0):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R < -80 (oversold) OR trend turns down (ADX falling below 20)
            elif curr_williams_r < -80.0 or (trending and curr_adx < 20.0):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals