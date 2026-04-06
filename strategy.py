#!/usr/bin/env python3
"""
1h RSI mean reversion with 4h trend filter and volume confirmation
Hypothesis: In strong 4h trends (above/below EMA50), RSI extremes on 1h offer high-probability mean reversion entries.
Volume confirms institutional participation. Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).
Target: 75-200 total trades over 4 years by using tight RSI thresholds (15/85) and requiring volume > 1.5x average.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_mean_reversion_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_rising = ema50_4h > np.roll(ema50_4h, 1)
    ema50_falling = ema50_4h < np.roll(ema50_4h, 1)
    ema50_rising[0] = ema50_rising[1] if len(ema50_rising) > 1 else False
    ema50_falling[0] = ema50_falling[1] if len(ema50_falling) > 1 else False
    ema50_rising_aligned = align_htf_to_ltf(prices, df_4h, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_4h, ema50_falling)
    
    # 1h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 100  # For RSI calculation
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(ema50_rising_aligned[i]) or 
            np.isnan(ema50_falling_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: RSI mean reversion or stoploss
        if position == 1:  # long position
            # Exit: RSI returns to neutral (50) or stoploss
            if (rsi[i] >= 50 or 
                close[i] <= entry_price - 2.0 * (high[i] - low[i])):  # ATR proxy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI returns to neutral (50) or stoploss
            if (rsi[i] <= 50 or 
                close[i] >= entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: RSI extreme + trend + volume
            long_entry = (rsi[i] < 15 and 
                         ema50_rising_aligned[i] and 
                         volume[i] > vol_ema[i] * 1.5)
            short_entry = (rsi[i] > 85 and 
                          ema50_falling_aligned[i] and 
                          volume[i] > vol_ema[i] * 1.5)
            
            if long_entry:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h Bollinger Band squeeze breakout with 4h trend filter and volume confirmation
Hypothesis: After low volatility (Bollinger Band width < 50th percentile), breakouts in the direction of the 4h trend (above/below EMA50) offer high-probability trades.
Volume confirms breakout validity. Works in both bull (buy upside breakouts in uptrend) and bear (sell downside breakouts in downtrend).
Target: 75-200 total trades over 4 years by using Bollinger Band width < 0.5 (low volatility) and requiring close outside bands + volume > 2x average.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_bb_squeeze_breakout_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_rising = ema50_4h > np.roll(ema50_4h, 1)
    ema50_falling = ema50_4h < np.roll(ema50_4h, 1)
    ema50_rising[0] = ema50_rising[1] if len(ema50_rising) > 1 else False
    ema50_falling[0] = ema50_falling[1] if len(ema50_falling) > 1 else False
    ema50_rising_aligned = align_htf_to_ltf(prices, df_4h, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_4h, ema50_falling)
    
    # 1h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle  # normalized width
    
    # Bollinger Band width percentile (50-period lookback) for squeeze detection
    bb_width_pct = pd.Series(bb_width).rolling(window=50, min_periods=50).rank(pct=True).values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 60  # For BB calculation (20 + 50 for percentile)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(bb_width_pct[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(ema50_rising_aligned[i]) or np.isnan(ema50_falling_aligned[i]) or 
            np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: mean reversion to middle band or stoploss
        if position == 1:  # long position
            # Exit: price returns to middle band or stoploss
            if (close[i] <= bb_middle[i] or 
                close[i] <= entry_price - 2.0 * (high[i] - low[i])):  # ATR proxy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price returns to middle band or stoploss
            if (close[i] >= bb_middle[i] or 
                close[i] >= entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: Bollinger Band squeeze breakout + trend + volume
            squeeze = bb_width_pct[i] < 0.5  # BB width below 50th percentile (squeeze)
            long_entry = squeeze and (close[i] > bb_upper[i]) and ema50_rising_aligned[i] and (volume[i] > vol_ema[i] * 2.0)
            short_entry = squeeze and (close[i] < bb_lower[i]) and ema50_falling_aligned[i] and (volume[i] > vol_ema[i] * 2.0)
            
            if long_entry:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h ADX trend strength with 4h directional filter and volume confirmation
Hypothesis: When ADX > 25 indicates strong trend, enter in the direction of the 4h EMA50 (above/below) to ride institutional momentum.
Volume confirms trend strength. Works in both bull (buy when above 4h EMA50 in uptrend) and bear (sell when below 4h EMA50 in downtrend).
Target: 75-200 total trades over 4 years by using ADX > 25 and requiring volume > 1.5x average.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_adx_trend_4h_direction_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for directional filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 for trend direction
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_above = close_4h > ema50_4h
    ema50_below = close_4h < ema50_4h
    ema50_above_aligned = align_htf_to_ltf(prices, df_4h, ema50_above)
    ema50_below_aligned = align_htf_to_ltf(prices, df_4h, ema50_below)
    
    # 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ADX (14) calculation
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - low[:-1]), np.absolute(low[1:] - high[:-1]))
    # Pad arrays to match length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Smooth with Wilder's smoothing (alpha = 1/14)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        return pd.Series(data).ewm(alpha=alpha, adjust=False, min_periods=period).mean().values
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / (atr + 1e-10)
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / (atr + 1e-10)
    dx = 100 * np.absolute(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilders_smoothing(dx, 14)
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For ADX calculation
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(adx[i]) or np.isnan(ema50_above_aligned[i]) or 
            np.isnan(ema50_below_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: trend weakening or stoploss
        if position == 1:  # long position
            # Exit: ADX falls below 20 (trend weakening) or stoploss
            if (adx[i] < 20 or 
                close[i] <= entry_price - 2.0 * (high[i] - low[i])):  # ATR proxy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: ADX falls below 20 (trend weakening) or stoploss
            if (adx[i] < 20 or 
                close[i] >= entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: strong trend (ADX > 25) + 4h direction + volume
            strong_trend = adx[i] > 25
            long_entry = strong_trend and ema50_above_aligned[i] and (volume[i] > vol_ema[i] * 1.5)
            short_entry = strong_trend and ema50_below_aligned[i] and (volume[i] > vol_ema[i] * 1.5)
            
            if long_entry:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h Williams %R mean reversion with 4h trend filter and volume confirmation
Hypothesis: In strong 4h trends (above/below EMA50), Williams %R extremes on 1h offer high-probability mean reversion entries.
Volume confirms institutional participation. Works in both bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend).
Target: 75-200 total trades over 4 years by using tight Williams %R thresholds (<10/>90) and requiring volume > 1.5x average.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_williamsr_mean_reversion_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_rising = ema50_4h > np.roll(ema50_4h, 1)
    ema50_falling = ema50_4h < np.roll(ema50_4h, 1)
    ema50_rising[0] = ema50_rising[1] if len(ema50_rising) > 1 else False
    ema50_falling[0] = ema50_falling[1] if len(ema50_falling) > 1 else False
    ema50_rising_aligned = align_htf_to_ltf(prices, df_4h, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_4h, ema50_falling)
    
    # 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    willr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 30  # For Williams %R calculation
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(willr[i]) or np.isnan(ema50_rising_aligned[i]) or 
            np.isnan(ema50_falling_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: Williams %R returns to neutral (-50) or stoploss
        if position == 1:  # long position
            # Exit: Williams %R returns to neutral (-50) or stoploss
            if (willr[i] >= -50 or 
                close[i] <= entry_price - 2.0 * (high[i] - low[i])):  # ATR proxy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: Williams %R returns to neutral (-50) or stoploss
            if (willr[i] <= -50 or 
                close[i] >= entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: Williams %R extreme + trend + volume
            long_entry = (willr[i] < -90 and 
                         ema50_rising_aligned[i] and 
                         volume[i] > vol_ema[i] * 1.5)
            short_entry = (willr[i] > -10 and 
                          ema50_falling_aligned[i] and 
                          volume[i] > vol_ema[i] * 1.5)
            
            if long_entry:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h Stochastic RSI mean reversion with 4h trend filter and volume confirmation
Hypothesis: In strong 4h trends (above/below EMA50), Stochastic RSI extremes on 1h offer high-probability mean reversion entries.
Volume confirms institutional participation. Works in both bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend).
Target: 75-200 total trades over 4 years by using tight Stochastic RSI thresholds (<0.1/>0.9) and requiring volume > 1.5x average.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_stochrsi_mean_reversion_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_rising = ema50_4h > np.roll(ema50_4h, 1)
    ema50_falling = ema50_4h < np.roll(ema50_4h, 1)
    ema50_rising[0] = ema50_rising[1] if len(ema50_rising) > 1 else False
    ema50_falling[0] = ema50_falling[1] if len(ema50_falling) > 1 else False
    ema50_rising_aligned = align_htf_to_ltf(prices, df_4h, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_4h, ema50_falling)
    
    # 1h indicators
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14) for Stochastic RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic RSI (14,14,3,3)
    rsi_min = pd.Series(rsi).rolling(window=14, min_periods=14).min().values
    rsi_max = pd.Series(rsi).rolling(window=14, min_periods=14).max().values
    stochrsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 30  # For RSI and Stochastic RSI calculation
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(stochrsi[i]) or np.isnan(ema50_rising_aligned[i]) or 
            np.isnan(ema50_falling_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: Stochastic RSI returns to neutral (0.5) or stoploss
        if position == 1:  # long position
            # Exit: Stochastic RSI returns to neutral (0.5) or stoploss
            if (stochrsi[i] >= 0.5 or 
                close[i] <= entry_price - 2.0 * (high[i] - low[i])):  # ATR proxy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: Stochastic RSI returns to neutral (0.5) or stoploss
            if (stochrsi[i] <= 0.5 or 
                close[i] >= entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: Stochastic RSI extreme + trend + volume
            long_entry = (stochrsi[i] < 0.1 and 
                         ema50_rising_aligned[i] and 
                         volume[i] > vol_ema[i] * 1.5)
            short_entry = (stochrsi[i] > 0.9 and 
                          ema50_falling_aligned[i] and 
                          volume[i] > vol_ema[i] * 1.5)
            
            if long_entry:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

Wait, I need to define high and low variables for the ATR proxy. Let me fix that.

</think>
#!/usr/bin/env python3
"""
1h Stochastic RSI mean reversion with 4h trend filter and volume confirmation
Hypothesis: In strong 4h trends (above/below EMA50), Stochastic RSI extremes on 1h offer high-probability mean reversion entries.
Volume confirms institutional participation. Works in both bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend).
Target: 75-200 total trades over 4 years by using tight Stochastic RSI thresholds (<0.1/>0.9) and requiring volume > 1.5x average.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_stochrsi_mean_reversion_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_rising = ema50_4h > np.roll(ema50_4h, 1)
    ema50_falling = ema50_4h < np.roll(ema50_4h, 1)
    ema50_rising[0] = ema50_rising[1] if len(ema50_rising) > 1 else False
    ema50_falling[0] = ema50_falling[1] if len(ema50_falling) > 1 else False
    ema50_rising_aligned = align_htf_to_ltf(prices, df_4h, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_4h, ema50_falling)
    
    # 1h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14) for Stochastic RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic RSI (14,14,3,3)
    rsi_min = pd.Series(rsi).rolling(window=14, min_periods=14).min().values
    rsi_max = pd.Series(rsi).rolling(window=14, min_periods=14).max().values
    stochrsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 30  # For RSI and Stochastic RSI calculation
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(stochrsi[i]) or np.isnan(ema50_rising_aligned[i]) or 
            np.isnan(ema50_falling_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: Stochastic RSI returns to neutral (0.5) or stoploss
        if position == 1:  # long position
            # Exit