#!/usr/bin/env python3
"""
12h Donchian Breakout + Volume + ADX Trend Filter
Hypothesis: Donchian(20) breakouts capture strong momentum moves. Volume confirms breakout strength,
and ADX > 25 filters for trending markets, avoiding false breakouts in ranging conditions.
Works in bull markets (breakout above upper band) and bear markets (breakout below lower band).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14365_12h_donchian20_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation on 1d
    adx_period = 14
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=adx_period, min_periods=adx_period).sum().values / pd.Series(atr_1d).rolling(window=adx_period, min_periods=adx_period).sum().values
    minus_di = 100 * pd.Series(minus_dm).rolling(window=adx_period, min_periods=adx_period).sum().values / pd.Series(atr_1d).rolling(window=adx_period, min_periods=adx_period).sum().values
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    # Align ADX to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_window = 20
    upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)  # Require 150% of average volume for breakout
    
    # ATR for stoploss (12h)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(donchian_window, adx_period) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR ADX < 20 (trend weakening) OR stoploss
            if (close[i] < lower[i] or adx_aligned[i] < 20 or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR ADX < 20 OR stoploss
            if (close[i] > upper[i] or adx_aligned[i] < 20 or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + ADX > 25 (trending)
            long_breakout = close[i] > upper[i]
            short_breakout = close[i] < lower[i]
            
            if long_breakout and vol_filter[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and vol_filter[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
12h Bollinger Band Squeeze Breakout + Volume + ADX Trend Filter
Hypothesis: Bollinger Band squeeze (low volatility) precedes explosive breakouts.
Volume confirms breakout strength, and ADX > 25 filters for trending markets.
Works in bull (breakout above upper band) and bear (breakout below lower band).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14365_12h_bb_squeeze_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation on 1d
    adx_period = 14
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=adx_period, min_periods=adx_period).sum().values / pd.Series(atr_1d).rolling(window=adx_period, min_periods=adx_period).sum().values
    minus_di = 100 * pd.Series(minus_dm).rolling(window=adx_period, min_periods=adx_period).sum().values / pd.Series(atr_1d).rolling(window=adx_period, min_periods=adx_period).sum().values
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    # Align ADX to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_window = 20
    bb_std = 2
    ma = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).mean().values
    std = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).std().values
    upper = ma + (std * bb_std)
    lower = ma - (std * bb_std)
    
    # Bollinger Band Width (squeeze indicator)
    bb_width = (upper - lower) / ma
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze = bb_width < (0.5 * bb_width_ma)  # Squeeze when width < 50% of 50-period average
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)  # Require 150% of average volume for breakout
    
    # ATR for stoploss (12h)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(bb_window, adx_period) + 50  # Need 50 for BB width MA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or
            np.isnan(squeeze[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below lower Bollinger Band OR ADX < 20 OR stoploss
            if (close[i] < lower[i] or adx_aligned[i] < 20 or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Bollinger Band OR ADX < 20 OR stoploss
            if (close[i] > upper[i] or adx_aligned[i] < 20 or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: BB squeeze breakout + volume + ADX > 25 (trending)
            long_breakout = close[i] > upper[i]
            short_breakout = close[i] < lower[i]
            
            if long_breakout and vol_filter[i] and adx_aligned[i] > 25 and squeeze[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and vol_filter[i] and adx_aligned[i] > 25 and squeeze[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
12h KAMA Trend + Volume + ADX Filter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market noise,
providing smooth trend following in trending markets and avoiding whipsaws in ranging markets.
Volume confirms trend strength, and ADX > 25 filters for trending conditions.
Works in bull (price above KAMA) and bear (price below KAMA).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14365_12h_kama_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation on 1d
    adx_period = 14
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=adx_period, min_periods=adx_period).sum().values / pd.Series(atr_1d).rolling(window=adx_period, min_periods=adx_period).sum().values
    minus_di = 100 * pd.Series(minus_dm).rolling(window=adx_period, min_periods=adx_period).sum().values / pd.Series(atr_1d).rolling(window=adx_period, min_periods=adx_period).sum().values
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    # Align ADX to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 12h data
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA calculation
    kama_window = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, kama_window))  # |close - close[kama_window]|
    # Pad change array to match close length
    change = np.concatenate([np.full(kama_window, np.nan), change])
    
    # Volatility sum of absolute changes
    vol = np.sum(np.abs(np.diff(close, 1)).reshape(-1, 1), axis=1)  # This approach is wrong, let's do it properly
    
    # Correct KAMA calculation
    # Initialize arrays
    kama = np.full_like(close, np.nan)
    eff_ratio = np.full_like(close, np.nan)
    sc = np.full_like(close, np.nan)
    
    # Set first value
    kama[kama_window] = close[kama_window]
    
    for i in range(kama_window + 1, len(close)):
        # Calculate efficiency ratio
        price_change = np.abs(close[i] - close[i - kama_window])
        volatility = 0
        for j in range(i - kama_window + 1, i + 1):
            volatility += np.abs(close[j] - close[j-1])
        
        if volatility > 0:
            eff_ratio[i] = price_change / volatility
        else:
            eff_ratio[i] = 0
        
        # Calculate smoothing constant
        sc[i] = (eff_ratio[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Calculate KAMA
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Alternative simpler approach using pandas (but we need to avoid look-ahead)
    # Let's use a correct implementation
    close_series = pd.Series(close)
    # Direction = abs(close - close.shift(kama_window))
    direction = np.abs(close_series - close_series.shift(kama_window))
    # Volatility = sum of abs(close - close.shift(1)) over kama_window
    volatility = close_series.diff().abs().rolling(window=kama_window, min_periods=1).sum()
    # Efficiency Ratio = direction / volatility
    eff_ratio = direction / volatility
    # Smoothing Constant = [ER * (fast_sc - slow_sc) + slow_sc]^2
    sc = (eff_ratio * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA = previous KAMA + sc * (close - previous KAMA)
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)  # Require 150% of average volume
    
    # ATR for stoploss (12h)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(kama_window, adx_period) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price crosses below KAMA OR ADX < 20 OR stoploss
            if (close[i] < kama[i] or adx_aligned[i] < 20 or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above KAMA OR ADX < 20 OR stoploss
            if (close[i] > kama[i] or adx_aligned[i] < 20 or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price vs KAMA + volume + ADX > 25 (trending)
            if close[i] > kama[i] and vol_filter[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif close[i] < kama[i] and vol_filter[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
12h Donchian Breakout + Volume + ADX Trend Filter
Hypothesis: Donchian(20) breakouts capture strong momentum moves. Volume confirms breakout strength,
and ADX > 25 filters for trending markets, avoiding false breakouts in ranging conditions.
Works in bull markets (breakout above upper band) and bear markets (breakout below lower band).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14365_12h_donchian20_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation on 1d
    adx_period = 14
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=adx_period, min_periods=adx_period).sum().values / pd.Series(atr_1d).rolling(window=adx_period, min_periods=adx_period).sum().values
    minus_di = 100 * pd.Series(minus_dm).rolling(window=adx_period, min_periods=adx_period).sum().values / pd.Series(atr_1d).rolling(window=adx_period, min_periods=adx_period).sum().values
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    # Align ADX to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_window = 20
    upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)  # Require 150% of average volume for breakout
    
    # ATR for stoploss (12h)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(donchian_window, adx_period) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR ADX < 20 (trend weakening) OR stoploss
            if (close[i] < lower[i] or adx_aligned[i] < 20 or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR ADX < 20 OR stoploss
            if (close[i] > upper[i] or adx_aligned[i] < 20 or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + ADX > 25 (trending)
            long_breakout = close[i] > upper[i]
            short_breakout = close[i] < lower[i]
            
            if long_breakout and vol_filter[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and vol_filter[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
12h Donchian Breakout + Volume + ADX Trend Filter
Hypothesis: Donchian(20) breakouts capture strong momentum moves. Volume confirms breakout strength,
and ADX > 25 filters for trending markets, avoiding false breakouts in ranging conditions.
Works in bull markets (breakout above upper band) and bear markets (breakout below lower band).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14365_12h_donchian20_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ADX calculation on 1d
    adx_period = 14
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=adx_period, min_periods=adx_period).sum().values / pd.Series(atr_1d).rolling(window=adx_period, min_periods=adx_period).sum().values
    minus_di = 100 * pd.Series(minus_dm).rolling(window=adx_period, min_periods=adx_period).sum().values / pd.Series(atr_1d).