#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter and volume confirmation.
# Goes long when price breaks above 20-period high with volume > 1.5x average and 1d close > 1d open (bullish day).
# Goes short when price breaks below 20-period low with volume > 1.5x average and 1d close < 1d open (bearish day).
# Uses ATR-based stoploss (2x ATR). Target: 75-200 total trades over 4 years (19-50/year).

name = "6h_donchian20_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # 1d trend: bullish if close > open, bearish if close < open
    bullish_1d = close_1d > open_1d
    bearish_1d = close_1d < open_1d
    
    # Align 1d trend to 6h (shifted by 1 day for prior day's close)
    bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, bullish_1d.astype(float))
    bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, bearish_1d.astype(float))
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filters
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_strong = volume > (vol_ma * 1.5)  # Strong volume for breakouts
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(bullish_1d_aligned[i]) or np.isnan(bearish_1d_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian low or 1d trend turns bearish
            elif close[i] < low_min[i] or bearish_1d_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian high or 1d trend turns bullish
            elif close[i] > high_max[i] or bullish_1d_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and 1d trend filter
            if vol_strong[i]:
                # Long breakout: price breaks above Donchian high with strong volume and bullish 1d trend
                if close[i] > high_max[i] and bullish_1d_aligned[i] > 0.5:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price breaks below Donchian low with strong volume and bearish 1d trend
                elif close[i] < low_min[i] and bearish_1d_aligned[i] > 0.5:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter and volume confirmation.
# Goes long when price breaks above 20-period high with volume > 1.5x average and 1d close > 1d open (bullish day).
# Goes short when price breaks below 20-period low with volume > 1.5x average and 1d close < 1d open (bearish day).
# Uses ATR-based stoploss (2x ATR). Target: 75-200 total trades over 4 years (19-50/year).

name = "6h_donchian20_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # 1d trend: bullish if close > open, bearish if close < open
    bullish_1d = close_1d > open_1d
    bearish_1d = close_1d < open_1d
    
    # Align 1d trend to 6h (shifted by 1 day for prior day's close)
    bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, bullish_1d.astype(float))
    bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, bearish_1d.astype(float))
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filters
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_strong = volume > (vol_ma * 1.5)  # Strong volume for breakouts
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(bullish_1d_aligned[i]) or np.isnan(bearish_1d_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian low or 1d trend turns bearish
            elif close[i] < low_min[i] or bearish_1d_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian high or 1d trend turns bullish
            elif close[i] > high_max[i] or bullish_1d_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and 1d trend filter
            if vol_strong[i]:
                # Long breakout: price breaks above Donchian high with strong volume and bullish 1d trend
                if close[i] > high_max[i] and bullish_1d_aligned[i] > 0.5:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price breaks below Donchian low with strong volume and bearish 1d trend
                elif close[i] < low_min[i] and bearish_1d_aligned[i] > 0.5:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals