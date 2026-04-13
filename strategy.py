#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d trend filter (ADX>25) and volume confirmation (>1.5x 20-bar avg)
    # Long when: price breaks above Donchian upper band (20) AND 1d ADX > 25 AND volume > 1.5x 20-bar avg volume
    # Short when: price breaks below Donchian lower band (20) AND 1d ADX > 25 AND volume > 1.5x 20-bar avg volume
    # Exit when: price crosses Donchian midpoint OR 1d ADX < 20 (regime shift to ranging)
    # Uses discrete sizing (0.25) targeting 50-150 total trades over 4 years.
    # Donchian provides objective breakout levels; 1d ADX filters choppy markets; volume confirms validity.
    # Works in bull (trend continuation) and bear (strong directional moves only).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d timeframe for regime filter
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
        tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        up_move = high - np.concatenate([[high[0]], high[:-1]])
        down_move = np.concatenate([[low[0]], low[:-1]]) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Wilder's smoothing
        def wilders_smooth(data, period):
            result = np.full_like(data, np.nan, dtype=float)
            if len(data) < period:
                return result
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
            return result
        
        tr_period = wilders_smooth(tr, period)
        plus_di_period = wilders_smooth(plus_dm, period)
        minus_di_period = wilders_smooth(minus_dm, period)
        
        # Avoid division by zero
        divisor = tr_period.copy()
        divisor[divisor == 0] = 1e-10
        
        plus_di = 100 * (plus_di_period / divisor)
        minus_di = 100 * (minus_di_period / divisor)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilders_smooth(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d ADX to 6h timeframe (wait for completed 1d bar)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Donchian channels (20-period) on 6h
    def donchian_channels(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        mid = (upper + lower) / 2.0
        return upper, lower, mid
    
    donchian_upper, donchian_lower, donchian_mid = donchian_channels(high, low, 20)
    
    # Volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs previous bar's bands to avoid look-ahead)
        breakout_up = close[i] > donchian_upper[i-1]  # break above previous upper band
        breakout_down = close[i] < donchian_lower[i-1]  # break below previous lower band
        
        # 1d ADX regime filter: only trade when trending (ADX > 25)
        strong_trend = adx_1d_aligned[i] > 25
        ranging_market = adx_1d_aligned[i] < 20  # exit condition
        
        # Entry conditions with volume confirmation and trend filter
        long_entry = breakout_up and strong_trend and volume_confirmed[i] and position != 1
        short_entry = breakout_down and strong_trend and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (close[i] < donchian_mid[i] or ranging_market))
        exit_short = (position == -1 and (close[i] > donchian_mid[i] or ranging_market))
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_donchian_breakout_adx_volume_v1"
timeframe = "6h"
leverage = 1.0