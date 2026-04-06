#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout with 1-day ADX trend filter and 1-week volume confirmation.
# Breakouts capture momentum in both bull and bear markets. ADX ensures we only trade in trending markets to avoid false breakouts.
# Volume confirmation ensures institutional participation. Designed for 12h timeframe to target 50-150 trades over 4 years.
# Uses discrete position sizing (0.25) to minimize churn.

name = "12h_donchian20_1d_adx1w_vol_v1"
timeframe = "12h"
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
    
    # 1-day Donchian(20) for breakout signals
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channels: upper = max(high, 20), lower = min(low, 20)
    upper_1d = np.full(len(high_1d), np.nan)
    lower_1d = np.full(len(low_1d), np.nan)
    for i in range(19, len(high_1d)):
        upper_1d[i] = np.max(high_1d[i-19:i+1])
        lower_1d[i] = np.min(low_1d[i-19:i+1])
    
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # 1-week ADX(14) for trend strength filtering
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range and Directional Movement
    tr = np.full(len(close_1w), np.nan)
    dm_plus = np.full(len(close_1w), np.nan)
    dm_minus = np.full(len(close_1w), np.nan)
    
    if len(close_1w) > 1:
        tr[0] = high_1w[0] - low_1w[0]
        dm_plus[0] = 0
        dm_minus[0] = 0
        for i in range(1, len(close_1w)):
            tr[i] = max(high_1w[i] - low_1w[i],
                       abs(high_1w[i] - close_1w[i-1]),
                       abs(low_1w[i] - close_1w[i-1]))
            dm_plus[i] = max(high_1w[i] - high_1w[i-1], 0)
            dm_minus[i] = max(low_1w[i-1] - low_1w[i], 0)
    
    # Smoothed TR, DM+, DM-
    atr_1w = np.full(len(close_1w), np.nan)
    s_dm_plus = np.full(len(close_1w), np.nan)
    s_dm_minus = np.full(len(close_1w), np.nan)
    
    if len(close_1w) >= 14:
        atr_1w[13] = np.nansum(tr[1:14])
        s_dm_plus[13] = np.nansum(dm_plus[1:14])
        s_dm_minus[13] = np.nansum(dm_minus[1:14])
        for i in range(14, len(close_1w)):
            atr_1w[i] = atr_1w[i-1] - (atr_1w[i-1]/14) + tr[i]
            s_dm_plus[i] = s_dm_plus[i-1] - (s_dm_plus[i-1]/14) + dm_plus[i]
            s_dm_minus[i] = s_dm_minus[i-1] - (s_dm_minus[i-1]/14) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.full(len(close_1w), np.nan)
    di_minus = np.full(len(close_1w), np.nan)
    dx = np.full(len(close_1w), np.nan)
    
    for i in range(13, len(close_1w)):
        if atr_1w[i] != 0:
            di_plus[i] = 100 * s_dm_plus[i] / atr_1w[i]
            di_minus[i] = 100 * s_dm_minus[i] / atr_1w[i]
            if di_plus[i] + di_minus[i] != 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX calculation
    adx = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 27:
        dx_valid = dx[13:]
        if len(dx_valid) >= 14:
            adx[26] = np.nanmean(dx_valid[:14])
            for i in range(27, len(close_1w)):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # 1-week volume average for confirmation
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    for i in range(4, len(vol_1w)):
        vol_ma_1w[i] = np.mean(vol_1w[i-4:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(27, 19, 4)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.3x weekly average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.3
        
        # ADX filter: only trade when trending (ADX > 20)
        trending_market = adx_aligned[i] > 20
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian or stoploss
            if (close[i] < lower_aligned[i] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian or stoploss
            if (close[i] > upper_aligned[i] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries in trending markets
            if trending_market and volume_filter:
                # Long: price breaks above upper Donchian
                if close[i] > upper_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below lower Donchian
                elif close[i] < lower_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout with 1-day ADX trend filter and 1-week volume confirmation.
# Breakouts capture momentum in both bull and bear markets. ADX ensures we only trade in trending markets to avoid false breakouts.
# Volume confirmation ensures institutional participation. Designed for 12h timeframe to target 50-150 trades over 4 years.
# Uses discrete position sizing (0.25) to minimize churn.

name = "12h_donchian20_1d_adx1w_vol_v1"
timeframe = "12h"
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
    
    # 1-day Donchian(20) for breakout signals
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channels: upper = max(high, 20), lower = min(low, 20)
    upper_1d = np.full(len(high_1d), np.nan)
    lower_1d = np.full(len(low_1d), np.nan)
    for i in range(19, len(high_1d)):
        upper_1d[i] = np.max(high_1d[i-19:i+1])
        lower_1d[i] = np.min(low_1d[i-19:i+1])
    
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # 1-week ADX(14) for trend strength filtering
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range and Directional Movement
    tr = np.full(len(close_1w), np.nan)
    dm_plus = np.full(len(close_1w), np.nan)
    dm_minus = np.full(len(close_1w), np.nan)
    
    if len(close_1w) > 1:
        tr[0] = high_1w[0] - low_1w[0]
        dm_plus[0] = 0
        dm_minus[0] = 0
        for i in range(1, len(close_1w)):
            tr[i] = max(high_1w[i] - low_1w[i],
                       abs(high_1w[i] - close_1w[i-1]),
                       abs(low_1w[i] - close_1w[i-1]))
            dm_plus[i] = max(high_1w[i] - high_1w[i-1], 0)
            dm_minus[i] = max(low_1w[i-1] - low_1w[i], 0)
    
    # Smoothed TR, DM+, DM-
    atr_1w = np.full(len(close_1w), np.nan)
    s_dm_plus = np.full(len(close_1w), np.nan)
    s_dm_minus = np.full(len(close_1w), np.nan)
    
    if len(close_1w) >= 14:
        atr_1w[13] = np.nansum(tr[1:14])
        s_dm_plus[13] = np.nansum(dm_plus[1:14])
        s_dm_minus[13] = np.nansum(dm_minus[1:14])
        for i in range(14, len(close_1w)):
            atr_1w[i] = atr_1w[i-1] - (atr_1w[i-1]/14) + tr[i]
            s_dm_plus[i] = s_dm_plus[i-1] - (s_dm_plus[i-1]/14) + dm_plus[i]
            s_dm_minus[i] = s_dm_minus[i-1] - (s_dm_minus[i-1]/14) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.full(len(close_1w), np.nan)
    di_minus = np.full(len(close_1w), np.nan)
    dx = np.full(len(close_1w), np.nan)
    
    for i in range(13, len(close_1w)):
        if atr_1w[i] != 0:
            di_plus[i] = 100 * s_dm_plus[i] / atr_1w[i]
            di_minus[i] = 100 * s_dm_minus[i] / atr_1w[i]
            if di_plus[i] + di_minus[i] != 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX calculation
    adx = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 27:
        dx_valid = dx[13:]
        if len(dx_valid) >= 14:
            adx[26] = np.nanmean(dx_valid[:14])
            for i in range(27, len(close_1w)):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # 1-week volume average for confirmation
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    for i in range(4, len(vol_1w)):
        vol_ma_1w[i] = np.mean(vol_1w[i-4:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(27, 19, 4)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.3x weekly average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.3
        
        # ADX filter: only trade when trending (ADX > 20)
        trending_market = adx_aligned[i] > 20
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian or stoploss
            if (close[i] < lower_aligned[i] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian or stoploss
            if (close[i] > upper_aligned[i] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries in trending markets
            if trending_market and volume_filter:
                # Long: price breaks above upper Donchian
                if close[i] > upper_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below lower Donchian
                elif close[i] < lower_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals