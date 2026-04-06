#!/usr/bin/env python3
"""
6h Bollinger Band Breakout with 1d Trend Filter and Volume Confirmation
Hypothesis: Bollinger Band breakouts capture momentum bursts, filtered by 1d EMA200 trend direction and volume confirmation.
Works in bull markets (buy breakouts above upper band when price > 1d EMA200) and bear markets (sell breakdowns below lower band when price < 1d EMA200).
Uses Bollinger Bands (20,2) for volatility-based breakout detection. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_bb_breakout_1d_ema200_vol_v1"
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
    
    # Bollinger Bands (20,2)
    bb_period = 20
    bb_std = 2
    sma = np.full(n, np.nan)
    std = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(bb_period, n):
        sma[i] = np.mean(close[i-bb_period:i])
        std[i] = np.std(close[i-bb_period:i])
        upper[i] = sma[i] + bb_std * std[i]
        lower[i] = sma[i] - bb_std * std[i]
    
    # Get 1d data for trend filter (EMA200)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA200 on 1d close
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 198) / 200
    
    # Trend: above EMA200 = bullish, below = bearish
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    
    # Align trend to 6h timeframe
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Get volume data for confirmation
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align volume MA to 6h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(bb_period, 200)  # Need enough data for Bollinger Bands and EMA200
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x average 1d volume (scaled)
        # Scale 1d volume to 6h: approx 1/4 of 1d volume (since 4x 6h in 1d)
        vol_threshold = vol_ma_1d_aligned[i] / 4.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        session_filter = 8 <= hour <= 20
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes back inside Bollinger Bands OR against trend
            # Stoploss: price drops 2*ATR below entry (using Bollinger Band width as proxy)
            bb_width = upper[i] - lower[i]
            atr_proxy = bb_width  # Simple proxy for volatility
            if (close[i] < upper[i] and close[i] > lower[i]) or \
               trend_1d_aligned[i] == -1 or \
               close[i] < entry_price - 2.0 * atr_proxy:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes back inside Bollinger Bands OR against trend
            # Stoploss: price rises 2*ATR above entry
            bb_width = upper[i] - lower[i]
            atr_proxy = bb_width
            if (close[i] < upper[i] and close[i] > lower[i]) or \
               trend_1d_aligned[i] == 1 or \
               close[i] > entry_price + 2.0 * atr_proxy:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 6 bars flat
            if bars_since_entry >= 6:
                # Breakout entries: upper/lower with trend
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper band with bullish trend + volume + session
                if bull_breakout and trend_1d_aligned[i] == 1 and volume_filter and session_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower band with bearish trend + volume + session
                elif bear_breakout and trend_1d_aligned[i] == -1 and volume_filter and session_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bands_since_entry += 1
    
    return signals
</x<|endoftext|>I notice the code has a typo: 'bands_since_entry' should be 'bars_since_entry'. Let me fix that.

</think>
#!/usr/bin/env python3
"""
6h Bollinger Band Breakout with 1d Trend Filter and Volume Confirmation
Hypothesis: Bollinger Band breakouts capture momentum bursts, filtered by 1d EMA200 trend direction and volume confirmation.
Works in bull markets (buy breakouts above upper band when price > 1d EMA200) and bear markets (sell breakdowns below lower band when price < 1d EMA200).
Uses Bollinger Bands (20,2) for volatility-based breakout detection. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_bb_breakout_1d_ema200_vol_v1"
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
    
    # Bollinger Bands (20,2)
    bb_period = 20
    bb_std = 2
    sma = np.full(n, np.nan)
    std = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(bb_period, n):
        sma[i] = np.mean(close[i-bb_period:i])
        std[i] = np.std(close[i-bb_period:i])
        upper[i] = sma[i] + bb_std * std[i]
        lower[i] = sma[i] - bb_std * std[i]
    
    # Get 1d data for trend filter (EMA200)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA200 on 1d close
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 198) / 200
    
    # Trend: above EMA200 = bullish, below = bearish
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    
    # Align trend to 6h timeframe
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Get volume data for confirmation
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align volume MA to 6h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(bb_period, 200)  # Need enough data for Bollinger Bands and EMA200
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x average 1d volume (scaled)
        # Scale 1d volume to 6h: approx 1/4 of 1d volume (since 4x 6h in 1d)
        vol_threshold = vol_ma_1d_aligned[i] / 4.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        session_filter = 8 <= hour <= 20
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes back inside Bollinger Bands OR against trend
            # Stoploss: price drops 2*ATR below entry (using Bollinger Band width as proxy)
            bb_width = upper[i] - lower[i]
            atr_proxy = bb_width  # Simple proxy for volatility
            if (close[i] < upper[i] and close[i] > lower[i]) or \
               trend_1d_aligned[i] == -1 or \
               close[i] < entry_price - 2.0 * atr_proxy:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes back inside Bollinger Bands OR against trend
            # Stoploss: price rises 2*ATR above entry
            bb_width = upper[i] - lower[i]
            atr_proxy = bb_width
            if (close[i] < upper[i] and close[i] > lower[i]) or \
               trend_1d_aligned[i] == 1 or \
               close[i] > entry_price + 2.0 * atr_proxy:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 6 bars flat
            if bars_since_entry >= 6:
                # Breakout entries: upper/lower with trend
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Long: breakout above upper band with bullish trend + volume + session
                if bull_breakout and trend_1d_aligned[i] == 1 and volume_filter and session_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower band with bearish trend + volume + session
                elif bear_breakout and trend_1d_aligned[i] == -1 and volume_filter and session_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals
</x<|endoftext|>I notice the code has a typo: 'bands_since_entry' should be 'bars_since_entry'. Let me fix that.