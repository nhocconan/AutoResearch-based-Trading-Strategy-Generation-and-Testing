#!/usr/bin/env python3
"""
1h VWAP + Volume Profile + 4h Trend Filter
Hypothesis: VWAP acts as dynamic fair value with volume profile identifying value areas.
4h trend provides directional bias. Long when 4h trend up and price > VWAP + volume confirmation.
Short when 4h trend down and price < VWAP - volume confirmation.
Targets 60-150 total trades over 4 years by requiring 4h trend alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14354_1h_vwap_volume_profile_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # 4h trend: 20 EMA
    ema_4h = pd.Series(close_4h).ewm(span=20, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # VWAP calculation
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / vwap_denominator
    
    # Volume profile: high volume nodes
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Exit conditions
        if position == 1:  # long position
            if (not in_session or close[i] <= vwap[i] or close[i] <= entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if (not in_session or close[i] >= vwap[i] or close[i] >= entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Entry conditions: 4h trend + VWAP + volume
            trend_up = ema_4h_aligned[i] > ema_4h_aligned[i-1]
            trend_down = ema_4h_aligned[i] < ema_4h_aligned[i-1]
            price_above_vwap = close[i] > vwap[i]
            price_below_vwap = close[i] < vwap[i]
            volume_confirm = volume[i] > vol_ma[i]
            
            if in_session and trend_up and price_above_vwap and volume_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif in_session and trend_down and price_below_vwap and volume_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h Bollinger Bands + Volume + 4h Trend
Hypothesis: Bollinger Bands identify overbought/oversold conditions with mean reversion in ranging markets.
4h EMA provides trend filter to avoid counter-trend trades. Volume confirms breakouts.
Long when 4h uptrend, price touches lower BB with volume spike.
Short when 4h downtrend, price touches upper BB with volume spike.
Targets 60-150 total trades by requiring 4h trend alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14354_1h_bb_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # 4h trend: 34 EMA
    ema_4h = pd.Series(close_4h).ewm(span=34, min_periods=34).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2.0)
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(bb_period, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(sma[i]) or np.isnan(std[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Exit conditions
        if position == 1:  # long position
            if (not in_session or close[i] >= sma[i] or close[i] <= entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if (not in_session or close[i] <= sma[i] or close[i] >= entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Entry conditions: 4h trend + BB touch + volume
            trend_up = ema_4h_aligned[i] > ema_4h_aligned[i-1]
            trend_down = ema_4h_aligned[i] < ema_4h_aligned[i-1]
            price_at_lower_bb = close[i] <= lower_band[i]
            price_at_upper_bb = close[i] >= upper_band[i]
            volume_confirm = volume[i] > vol_ma[i]
            
            if in_session and trend_up and price_at_lower_bb and volume_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif in_session and trend_down and price_at_upper_bb and volume_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h Donchian Channel Breakout + Volume + 4h Trend Filter
Hypothesis: Donchian channels breakouts capture momentum with volume confirmation.
4h EMA filter ensures trades align with higher timeframe trend to avoid whipsaws.
Long when price breaks above 20-period Donchian high with volume and 4h uptrend.
Short when price breaks below 20-period Donchian low with volume and 4h downtrend.
Targets 60-150 total trades by requiring multiple confirmations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14354_1h_donchian_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # 4h trend: 50 EMA
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20 period)
    dc_period = 20
    dc_high = pd.Series(high).rolling(window=dc_period, min_periods=dc_period).max().values
    dc_low = pd.Series(low).rolling(window=dc_period, min_periods=dc_period).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(dc_period, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Exit conditions
        if position == 1:  # long position
            if (not in_session or close[i] <= dc_high[i-1] or close[i] <= entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if (not in_session or close[i] >= dc_low[i-1] or close[i] >= entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Entry conditions: 4h trend + Donchian breakout + volume
            trend_up = ema_4h_aligned[i] > ema_4h_aligned[i-1]
            trend_down = ema_4h_aligned[i] < ema_4h_aligned[i-1]
            breakout_up = close[i] > dc_high[i-1]
            breakout_down = close[i] < dc_low[i-1]
            volume_confirm = volume[i] > vol_ma[i]
            
            if in_session and trend_up and breakout_up and volume_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif in_session and trend_down and breakout_down and volume_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h Keltner Channel + RSI + 4h Trend Filter
Hypothesis: Keltner channels identify dynamic support/resistance with ATR-based bands.
RSI(14) provides momentum confirmation. 4h EMA filters for trend alignment.
Long when 4h uptrend, price touches lower KC with RSI < 40.
Short when 4h downtrend, price touches upper KC with RSI > 60.
Targets 60-150 total trades by requiring multiple confirmations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14354_1h_keltner_rsi_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # 4h trend: 50 EMA
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Keltner Channel (20, 2.0 ATR)
    kc_period = 20
    kc_multiplier = 2.0
    ema = pd.Series(close).ewm(span=kc_period, min_periods=kc_period).mean().values
    
    # ATR for KC bands
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    kc_upper = ema + (kc_multiplier * atr)
    kc_lower = ema - (kc_multiplier * atr)
    
    # RSI (14)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(kc_period, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Exit conditions
        if position == 1:  # long position
            if (not in_session or close[i] >= ema[i] or close[i] <= entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if (not in_session or close[i] <= ema[i] or close[i] >= entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Entry conditions: 4h trend + KC touch + RSI + volume
            trend_up = ema_4h_aligned[i] > ema_4h_aligned[i-1]
            trend_down = ema_4h_aligned[i] < ema_4h_aligned[i-1]
            price_at_lower_kc = close[i] <= kc_lower[i]
            price_at_upper_kc = close[i] >= kc_upper[i]
            rsi_oversold = rsi[i] < 40
            rsi_overbought = rsi[i] > 60
            volume_confirm = volume[i] > vol_ma[i]
            
            if in_session and trend_up and price_at_lower_kc and rsi_oversold and volume_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif in_session and trend_down and price_at_upper_kc and rsi_overbought and volume_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h Supertrend + Volume + 4h Trend Filter
Hypothesis: Supertrend provides dynamic trend following with ATR-based bands.
4h EMA filter ensures alignment with higher timeframe trend.
Volume confirms the strength of the trend.
Long when 4h uptrend, Supertrend flips to bullish with volume.
Short when 4h downtrend, Supertrend flips to bearish with volume.
Targets 60-150 total trades by requiring trend alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14354_1h_supertrend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # 4h trend: 89 EMA (longer term for stronger filter)
    ema_4h = pd.Series(close_4h).ewm(span=89, min_periods=89).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Supertrend (10, 3.0)
    atr_period = 10
    atr_multiplier = 3.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (atr_multiplier * atr)
    lower_band = hl2 - (atr_multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    # Calculate Supertrend
    for i in range(1, n):
        # Upper Band
        if upper_band[i] < upper_band[i-1] or close[i-1] > upper_band[i-1]:
            upper_band[i] = upper_band[i]
        else:
            upper_band[i] = upper_band[i-1]
        
        # Lower Band
        if lower_band[i] > lower_band[i-1] or close[i-1] < lower_band[i-1]:
            lower_band[i] = lower_band[i]
        else:
            lower_band[i] = lower_band[i-1]
        
        # Supertrend and Direction
        if supertrend[i-1] == upper_band[i-1]:
            if close[i] <= upper_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
        else:
            if close[i] >= lower_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(atr_period, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(supertrend[i]) or np.isnan(direction[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Exit conditions
        if position == 1:  # long position
            if (not in_session or direction[i] == -1 or close[i] <= entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if (not in_session or direction[i] == 1 or close[i] >= entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Entry conditions: 4h trend + Supertrend flip + volume
            trend_up = ema_4h_aligned[i] > ema_4h_aligned[i-1]
            trend_down = ema_4h_aligned[i] < ema_4h_aligned[i-1]
            st_flip_bullish = (direction[i] == 1 and direction[i-1] == -1)
            st_flip_bearish = (direction[i] == -1 and direction[i-1] == 1)
            volume_confirm = volume[i] > vol_ma[i]
            
            if in_session and trend_up and st_flip_bullish and volume_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif in_session and trend_down and st_flip_bearish and volume_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h Adaptive RSI + Bollinger Bands + 4h Trend Filter
Hypothesis: Adaptive RSI adjusts sensitivity based on market volatility.
Bollinger Bands identify volatility extremes. 4h EMA filters for trend alignment.
Long when 4h uptrend, adaptive RSI < 30, price touches lower BB.
Short when 4h downtrend, adaptive RSI > 70, price touches upper BB.
Targets 60-150 total trades by requiring multiple confirmations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14354_1h_adaptive_rsi_bb_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # 4h trend: 34 EMA
    ema_4h = pd.Series(close_4h).ewm(span=34, min_periods=34).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2.0)
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    
    # Adaptive RSI (14) - adjusts based on volatility
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Volatility adaptive smoothing
    vol_factor = pd.Series(std).rolling(window=10, min_periods=1).mean().values
    alpha = 2 / (rsi_period + 1) * (1 + vol_factor / (np.mean(vol_factor) + 1e-10))
    
    avg_gain = pd.Series(gain).ewm(alpha=alpha, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=alpha, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(bb_period, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(sma[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Exit conditions
        if position == 1:  # long position
            if (not in_session or close[i] >= sma[i] or close[i] <= entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if (not in_session or close[i] <= sma[i] or close[i] >= entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Entry conditions: 4h trend + adaptive RSI + BB touch + volume
            trend_up = ema_4h_aligned[i] > ema_4h_aligned[i-1]
            trend_down = ema_4h_aligned[i] < ema_4h_aligned[i-1]
            price_at_lower_bb = close[i] <= lower_band[i]
            price_at_upper_bb = close[i] >= upper_band[i]
            rsi_oversold = rsi[i] < 30
            rsi_overbought = rsi[i] > 70
            volume_confirm = volume[i] > vol_ma[i]
            
            if in_session and trend_up and price_at_lower_bb and rsi_oversold and volume_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif in_session and trend_down and price_at_upper_bb and rsi_overbought and volume_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1h Volume Weighted MACD + 4h Trend Filter
Hypothesis: Volume Weighted MACD improves standard MACD by incorporating volume.
4h EMA provides trend filter to avoid counter-trend trades.
Long when 4h uptrend, VW-MACD crosses above signal line with volume confirmation.
Short when 4h downtrend, VW-MACD crosses below signal line with volume confirmation.
Targets 60-150 total trades by requiring trend alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14354_1h_vw_macd_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return