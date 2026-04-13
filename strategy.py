#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout from 1d + volume spike + ADX regime filter
    # Long when: price breaks above Camarilla H3 (1d) AND ADX > 25 AND volume > 2.0x 20-bar avg volume
    # Short when: price breaks below Camarilla L3 (1d) AND ADX > 25 AND volume > 2.0x 20-bar avg volume
    # Exit when: price crosses Camarilla pivot point (PP) OR ADX < 20 (regime change to ranging)
    # Uses discrete sizing (0.25) targeting 75-200 total trades over 4 years.
    # Camarilla levels provide precise support/resistance; ADX filters ranging markets;
    # Volume spike confirms breakout validity. Works in bull (trend continuation) and bear (strong moves only).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # PP = (H + L + C) / 3
    # H3 = PP + (H - L) * 1.1 / 4
    # L3 = PP - (H - L) * 1.1 / 4
    PP_1d = (high_1d + low_1d + close_1d) / 3.0
    H3_1d = PP_1d + (high_1d - low_1d) * 1.1 / 4.0
    L3_1d = PP_1d - (high_1d - low_1d) * 1.1 / 4.0
    
    # Calculate ADX(14) on 4h timeframe for regime filter
    # ADX requires +DI, -DI, and TR
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    # Pad to match length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    tr_period = wilders_smoothing(tr, period_adx)
    plus_di_period = wilders_smoothing(plus_dm, period_adx)
    minus_di_period = wilders_smoothing(minus_dm, period_adx)
    
    # Avoid division by zero
    divisor = tr_period
    divisor[divisor == 0] = 1e-10
    
    plus_di = 100 * (plus_di_period / divisor)
    minus_di = 100 * (minus_di_period / divisor)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period_adx)
    
    # Align HTF indicators to 4h timeframe (wait for completed 1d bar)
    PP_1d_aligned = align_htf_to_ltf(prices, df_1d, PP_1d)
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    
    # Calculate volume confirmation: volume > 2.0x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(PP_1d_aligned[i]) or np.isnan(H3_1d_aligned[i]) or np.isnan(L3_1d_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions (using previous bar's levels to avoid look-ahead)
        breakout_up = close[i] > H3_1d_aligned[i-1]  # break above previous H3
        breakout_down = close[i] < L3_1d_aligned[i-1]  # break below previous L3
        
        # ADX regime filter: only trade when trending (ADX > 25)
        strong_trend = adx[i] > 25
        ranging_market = adx[i] < 20  # exit condition
        
        # Entry conditions with volume confirmation and trend filter
        long_entry = breakout_up and strong_trend and volume_confirmed[i] and position != 1
        short_entry = breakout_down and strong_trend and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (close[i] < PP_1d_aligned[i] or ranging_market))
        exit_short = (position == -1 and (close[i] > PP_1d_aligned[i] or ranging_market))
        
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

name = "4h_1d_camarilla_breakout_adx_volume_v1"
timeframe = "4h"
leverage = 1.0