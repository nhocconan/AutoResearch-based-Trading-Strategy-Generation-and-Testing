#!/usr/bin/env python3
"""
6h Bollinger Band Squeeze with 1d MACD Trend Filter
Hypothesis: Bollinger Band squeeze identifies low volatility periods preceding breakouts.
Trades are taken in the direction of the 1d MACD trend (bullish or bearish).
Breakouts are confirmed by volume expansion.
Works in both bull and bear markets as it captures volatility breakouts in the prevailing trend direction.
Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14451_6h_bb_squeeze_macd_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for MACD trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d MACD (12, 26, 9)
    ema12 = pd.Series(close_1d).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close_1d).ewm(span=26, adjust=False).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
    macd_trend = macd_line - signal_line  # Positive = bullish, Negative = bearish
    
    # Align MACD trend to 6h timeframe
    macd_trend_aligned = align_htf_to_ltf(prices, df_1d, macd_trend)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = ma + (bb_std * std_dev)
    lower_band = ma - (bb_std * std_dev)
    bb_width = upper_band - lower_band
    
    # Bollinger Band Squeeze: width below 20-period average width
    avg_width = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < (0.8 * avg_width)  # Squeeze when width < 80% of average
    
    # Volume filter: require volume above average for breakout confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.2 * vol_ma)  # Volume 20% above average
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(bb_period, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ma[i]) or np.isnan(std_dev[i]) or np.isnan(avg_width[i]) or
            np.isnan(vol_ma[i]) or np.isnan(macd_trend_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below middle band OR MACD turns bearish OR stoploss
            if (close[i] <= ma[i] or macd_trend_aligned[i] < 0 or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above middle band OR MACD turns bullish OR stoploss
            if (close[i] >= ma[i] or macd_trend_aligned[i] > 0 or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Bollinger Band breakout + MACD trend + volume
            long_breakout = close[i] > upper_band[i]
            short_breakout = close[i] < lower_band[i]
            
            # Only take breakouts in direction of 1d MACD trend
            long_setup = long_breakout and (macd_trend_aligned[i] > 0) and vol_filter[i] and squeeze[i-1]
            short_setup = short_breakout and (macd_trend_aligned[i] < 0) and vol_filter[i] and squeeze[i-1]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Bollinger Band Squeeze with 1d MACD Trend Filter
Hypothesis: Bollinger Band squeeze identifies low volatility periods preceding breakouts.
Trades are taken in the direction of the 1d MACD trend (bullish or bearish).
Breakouts are confirmed by volume expansion.
Works in both bull and bear markets as it captures volatility breakouts in the prevailing trend direction.
Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14451_6h_bb_squeeze_macd_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for MACD trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d MACD (12, 26, 9)
    ema12 = pd.Series(close_1d).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close_1d).ewm(span=26, adjust=False).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
    macd_trend = macd_line - signal_line  # Positive = bullish, Negative = bearish
    
    # Align MACD trend to 6h timeframe
    macd_trend_aligned = align_htf_to_ltf(prices, df_1d, macd_trend)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = ma + (bb_std * std_dev)
    lower_band = ma - (bb_std * std_dev)
    bb_width = upper_band - lower_band
    
    # Bollinger Band Squeeze: width below 20-period average width
    avg_width = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < (0.8 * avg_width)  # Squeeze when width < 80% of average
    
    # Volume filter: require volume above average for breakout confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.2 * vol_ma)  # Volume 20% above average
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(bb_period, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ma[i]) or np.isnan(std_dev[i]) or np.isnan(avg_width[i]) or
            np.isnan(vol_ma[i]) or np.isnan(macd_trend_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below middle band OR MACD turns bearish OR stoploss
            if (close[i] <= ma[i] or macd_trend_aligned[i] < 0 or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above middle band OR MACD turns bullish OR stoploss
            if (close[i] >= ma[i] or macd_trend_aligned[i] > 0 or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Bollinger Band breakout + MACD trend + volume
            long_breakout = close[i] > upper_band[i]
            short_breakout = close[i] < lower_band[i]
            
            # Only take breakouts in direction of 1d MACD trend
            long_setup = long_breakout and (macd_trend_aligned[i] > 0) and vol_filter[i] and squeeze[i-1]
            short_setup = short_breakout and (macd_trend_aligned[i] < 0) and vol_filter[i] and squeeze[i-1]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>