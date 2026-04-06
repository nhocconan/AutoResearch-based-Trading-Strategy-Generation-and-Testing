#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian(20) breakout with 4h ADX trend filter and 1d EMA200 trend filter
# Long when price breaks above Donchian(20) high + ADX > 25 + close > EMA200
# Short when price breaks below Donchian(20) low + ADX > 25 + close < EMA200
# Uses 4h ADX for trend strength and 1d EMA200 for trend direction
# Target: 80-120 total trades over 4 years with controlled risk
# ATR-based stoploss to limit drawdown and reduce whipsaw

name = "1h_donchian20_4h_adx_1d_ema200_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = 0  # First value has no previous close
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smooth TR, DM+, DM- using Wilder's smoothing (EMA with alpha=1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            for i in range(len(data)):
                if np.isnan(result[i-1]) if i > 0 else True:
                    result[i] = data[i]
                else:
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        tr_smooth = wilder_smooth(tr, period)
        dm_plus_smooth = wilder_smooth(dm_plus, period)
        dm_minus_smooth = wilder_smooth(dm_minus, period)
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / tr_smooth
        di_minus = 100 * dm_minus_smooth / tr_smooth
        
        # DX and ADX
        dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
        dx = np.where((di_plus + di_minus) == 0, 0, dx)
        
        adx = wilder_smooth(dx, period)
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    
    # Align 4h ADX to 1h timeframe
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # EMA200 calculation
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align 1d EMA200 to 1h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Donchian channels (20-period) - calculated on 1h data
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Apply session filter (08-20 UTC)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: ADX weak (<20) or trend reversal (close < EMA200) or Donchian breakdown
            elif adx_4h_aligned[i] < 20 or close[i] < ema200_1d_aligned[i] or close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: ADX weak (<20) or trend reversal (close > EMA200) or Donchian breakout
            elif adx_4h_aligned[i] < 20 or close[i] > ema200_1d_aligned[i] or close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with trend filters
            # Long: break above Donchian upper + strong trend (ADX>25) + uptrend (close > EMA200)
            if (in_session and
                close[i] > donchian_upper[i] and 
                adx_4h_aligned[i] > 25 and
                close[i] > ema200_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: break below Donchian lower + strong trend (ADX>25) + downtrend (close < EMA200)
            elif (in_session and
                  close[i] < donchian_lower[i] and 
                  adx_4h_aligned[i] > 25 and
                  close[i] < ema200_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals