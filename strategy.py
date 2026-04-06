#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA trend filter, volume confirmation, and ATR stoploss
# Enter long when: price breaks above Donchian high(20), close > 1d EMA(50), volume > 1.5x avg volume
# Enter short when: price breaks below Donchian low(20), close < 1d EMA(50), volume > 1.5x avg volume
# Exit when price moves 2x ATR against position or opposite Donchian break occurs
# Targets 80-150 trades over 4 years with controlled risk

name = "4h_donchian20_1dema_vol_atrstop_v2"
timeframe = "4h"
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
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # ATR(14) for stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = tr1[0]  # First bar: no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: stoploss hit OR price breaks below Donchian low(20)
            if close[i] < entry_price - 2.0 * atr[i] or close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: stoploss hit OR price breaks above Donchian high(20)
            if close[i] > entry_price + 2.0 * atr[i] or close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian break + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > high_20[i] and close[i] > ema_50_aligned[i]:
                    # Bullish breakout above 1d EMA
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < low_20[i] and close[i] < ema_50_aligned[i]:
                    # Bearish breakout below 1d EMA
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA trend filter, volume confirmation, and ATR stoploss
# Enter long when: price breaks above Donchian high(20), close > 1d EMA(50), volume > 1.5x avg volume
# Enter short when: price breaks below Donchian low(20), close < 1d EMA(50), volume > 1.5x avg volume
# Exit when price moves 2x ATR against position or opposite Donchian break occurs
# Targets 80-150 trades over 4 years with controlled risk

name = "4h_donchian20_1dema_vol_atrstop_v2"
timeframe = "4h"
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
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # ATR(14) for stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = tr1[0]  # First bar: no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: stoploss hit OR price breaks below Donchian low(20)
            if close[i] < entry_price - 2.0 * atr[i] or close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: stoploss hit OR price breaks above Donchian high(20)
            if close[i] > entry_price + 2.0 * atr[i] or close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian break + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > high_20[i] and close[i] > ema_50_aligned[i]:
                    # Bullish breakout above 1d EMA
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < low_20[i] and close[i] < ema_50_aligned[i]:
                    # Bearish breakout below 1d EMA
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals