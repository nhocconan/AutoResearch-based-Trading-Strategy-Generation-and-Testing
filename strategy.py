#!/usr/bin/env python3
"""
6h RSI(2) Extreme + 1d Trend Filter + Volume Confirmation
Hypothesis: RSI(2) identifies extreme short-term overbought/oversold conditions. 
In strong trends (1d EMA50), these extremes often precede continuations rather than reversals.
Volume confirms institutional participation. Works in bull (buy RSI<10 in uptrend) and 
bear (sell RSI>90 in downtrend). Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_ata, align_htf_to_ltf

name = "6h_rsi2_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(2) on 6h close
    rsi = np.full(n, 50.0)  # Start at neutral
    if n >= 3:
        # Calculate price changes
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        
        # Wilder's smoothing
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        avg_gain[2] = np.mean(gain[1:3])  # First 2-period average
        avg_loss[2] = np.mean(loss[1:3])
        
        for i in range(3, n):
            avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2
            avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        rsi[:2] = 50.0  # Not enough data
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 on 1d close
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 48) / 50
    
    # 1d trend: above EMA50 = bullish, below = bearish
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    
    # Align 1d trend to 6h timeframe
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Get 1d data for volume confirmation
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
    start = 50  # Need enough data for RSI and EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(trend_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x 1d average volume (scaled)
        # Scale 1d volume to 6h: approx 1/4 of 1d volume (since 4x 6h in 1d)
        vol_threshold = vol_ma_1d_aligned[i] / 4.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: RSI returns to neutral (>50) OR against 1d trend
            # Stoploss: price drops 2*ATR below entry (using 6-period ATR approximation)
            price_change = abs(close[i] - close[i-1]) if i > 0 else 0
            atr_approx = price_change * 6  # Rough 6-period ATR
            if (rsi[i] > 50 or
                trend_1d_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr_approx):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: RSI returns to neutral (<50) OR against 1d trend
            # Stoploss: price rises 2*ATR above entry
            price_change = abs(close[i] - close[i-1]) if i > 0 else 0
            atr_approx = price_change * 6
            if (rsi[i] < 50 or
                trend_1d_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr_approx):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 8 bars flat
            if bars_since_entry >= 8:
                # Extreme RSI entries
                rsi_oversold = rsi[i] < 10
                rsi_overbought = rsi[i] > 90
                
                # Long: RSI extremely oversold with bullish 1d trend + volume
                if rsi_oversold and trend_1d_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: RSI extremely overbought with bearish 1d trend + volume
                elif rsi_overbought and trend_1d_aligned[i] == -1 and volume_filter:
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

</think>
#!/usr/bin/env python3
"""
6h RSI(2) Extreme + 1d Trend Filter + Volume Confirmation
Hypothesis: RSI(2) identifies extreme short-term overbought/oversold conditions. 
In strong trends (1d EMA50), these extremes often precede continuations rather than reversals.
Volume confirms institutional participation. Works in bull (buy RSI<10 in uptrend) and 
bear (sell RSI>90 in downtrend). Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_rsi2_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(2) on 6h close
    rsi = np.full(n, 50.0)  # Start at neutral
    if n >= 3:
        # Calculate price changes
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        
        # Wilder's smoothing
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        avg_gain[2] = np.mean(gain[1:3])  # First 2-period average
        avg_loss[2] = np.mean(loss[1:3])
        
        for i in range(3, n):
            avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2
            avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        rsi[:2] = 50.0  # Not enough data
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 on 1d close
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 48) / 50
    
    # 1d trend: above EMA50 = bullish, below = bearish
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    
    # Align 1d trend to 6h timeframe
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Get 1d data for volume confirmation
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
    start = 50  # Need enough data for RSI and EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(trend_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x 1d average volume (scaled)
        # Scale 1d volume to 6h: approx 1/4 of 1d volume (since 4x 6h in 1d)
        vol_threshold = vol_ma_1d_aligned[i] / 4.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: RSI returns to neutral (>50) OR against 1d trend
            # Stoploss: price drops 2*ATR below entry (using 6-period ATR approximation)
            price_change = abs(close[i] - close[i-1]) if i > 0 else 0
            atr_approx = price_change * 6  # Rough 6-period ATR
            if (rsi[i] > 50 or
                trend_1d_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr_approx):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: RSI returns to neutral (<50) OR against 1d trend
            # Stoploss: price rises 2*ATR above entry
            price_change = abs(close[i] - close[i-1]) if i > 0 else 0
            atr_approx = price_change * 6
            if (rsi[i] < 50 or
                trend_1d_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr_approx):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 8 bars flat
            if bars_since_entry >= 8:
                # Extreme RSI entries
                rsi_oversold = rsi[i] < 10
                rsi_overbought = rsi[i] > 90
                
                # Long: RSI extremely oversold with bullish 1d trend + volume
                if rsi_oversold and trend_1d_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: RSI extremely overbought with bearish 1d trend + volume
                elif rsi_overbought and trend_1d_aligned[i] == -1 and volume_filter:
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