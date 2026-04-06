#!/usr/bin/env python3
"""
6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation
Hypothesis: Breakouts from Donchian channels on 6h, aligned with weekly pivot direction and volume spikes,
capture significant moves in both bull and bear markets. Weekly pivot provides directional bias from higher timeframe,
while volume confirms institutional participation. Designed for 6h timeframe with target of 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 6h and weekly data for pivot and trend (once before loop)
    df_6h = get_htf_data(prices, '6h')
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly pivot levels (using prior week's data)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate pivot points: P = (H+L+C)/3
    pivot = (high_weekly + low_weekly + close_weekly) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1 = 2 * pivot - low_weekly
    s1 = 2 * pivot - high_weekly
    # R2 = P + (H - L), S2 = P - (H - L)
    r2 = pivot + (high_weekly - low_weekly)
    s2 = pivot - (high_weekly - low_weekly)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    r3 = high_weekly + 2 * (pivot - low_weekly)
    s3 = low_weekly - 2 * (high_weekly - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian channel (20 periods)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # 6h volume filter (20-period average)
    vol_ma = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:  # 20 periods minimum
            vol_ma[i] = vol_sum / 20.0
    
    # Volatility filter: ATR(14) to avoid choppy markets
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = np.full(n, np.nan)
    atr_sum = 0.0
    for i in range(n):
        atr_sum += tr[i]
        if i >= 14:
            atr_sum -= tr[i-14]
        if i >= 13:  # 14 periods minimum
            atr[i] = atr_sum / 14.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of lookbacks)
    start = max(20, 20, 14)  # Donchian(20), Vol(20), ATR(14)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below Donchian low OR stoploss
            if (close[i] <= lowest_low[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian high OR stoploss
            if (close[i] >= highest_high[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Determine weekly pivot bias: price relative to pivot
            # Bullish bias: price above pivot, bearish: below pivot
            bullish_bias = close[i] > pivot_aligned[i]
            bearish_bias = close[i] < pivot_aligned[i]
            
            # Volume filter: current volume > 1.5 * 20-period average
            vol_filter = volume[i] > (1.5 * vol_ma[i])
            
            # Look for entries: Donchian breakout with pivot bias and volume
            long_breakout = (close[i] > highest_high[i] and bullish_bias and vol_filter)
            short_breakout = (close[i] < lowest_low[i] and bearish_bias and vol_filter)
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Camarilla Pivot Reversal with Volume Spike and Volatility Filter
Hypothesis: Camarilla pivot levels from 1d provide high-probability reversal points.
Trades fade at R3/S3 (strong reversal zones) and breakout continuation at R4/S4.
Volume spike confirms institutional interest, volatility filter avoids chop.
Designed for 6h timeframe targeting 50-150 trades over 4 years.
Works in bull/bear via mean reversion at extremes and trend continuation on breaks.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_reversal_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Prior day's OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels
    # Pivot = (H+L+C)/3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # Levels: R4 = C + (range * 1.1/2), R3 = C + (range * 1.1/4)
    #        S3 = C - (range * 1.1/4), S4 = C - (range * 1.1/2)
    r4 = close_1d + (range_1d * 1.1 / 2.0)
    r3 = close_1d + (range_1d * 1.1 / 4.0)
    s3 = close_1d - (range_1d * 1.1 / 4.0)
    s4 = close_1d - (range_1d * 1.1 / 2.0)
    
    # Align 1d Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h volatility filter: ATR(10) to identify trending vs choppy
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = np.full(n, np.nan)
    atr_sum = 0.0
    for i in range(n):
        atr_sum += tr[i]
        if i >= 10:
            atr_sum -= tr[i-10]
        if i >= 9:  # 10 periods minimum
            atr[i] = atr_sum / 10.0
    
    # 6h volume spike detector: current volume > 2.0 * 20-period average
    vol_ma = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:  # 20 periods minimum
            vol_ma[i] = vol_sum / 20.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For volume MA(20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: profit target at R3 or stoploss
            if (close[i] >= r3_aligned[i] or
                close[i] <= entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: profit target at S3 or stoploss
            if (close[i] <= s3_aligned[i] or
                close[i] >= entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Volume filter: spike indicates participation
            vol_filter = volume[i] > (2.0 * vol_ma[i])
            
            # Fade at R3/S3 (reversal zones)
            fade_long = (close[i] <= s3_aligned[i] and vol_filter)
            fade_short = (close[i] >= r3_aligned[i] and vol_filter)
            
            # Breakout continuation at R4/S4 (trend continuation)
            breakout_long = (close[i] >= r4_aligned[i] and vol_filter)
            breakout_short = (close[i] <= s4_aligned[i] and vol_filter)
            
            if fade_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif fade_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            elif breakout_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif breakout_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Elder Ray Index with 12h Trend Filter and Volume Confirmation
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) 
measures bull/bear strength relative to EMA13. Combined with 12h EMA trend filter
and volume confirmation, it captures strong directional moves while avoiding 
whipsaws. Works in bull/bear via symmetry of bull/bear power.
Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_12h_trend_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    # Load 6h and 12h data (once before loop)
    df_6h = get_htf_data(prices, '6h')
    df_12h = get_htf_data(prices, '12h')
    
    # 6h EMA13 for Elder Ray calculation
    close_6h = df_6h['close'].values
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False).mean().values
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components
    bull_power = high - ema13_6h  # Strength of bulls
    bear_power = ema13_6h - low   # Strength of bears
    
    # 6h volume filter (20-period average)
    vol_ma = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:  # 20 periods minimum
            vol_ma[i] = vol_sum / 20.0
    
    # 6h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = np.full(n, np.nan)
    atr_sum = 0.0
    for i in range(n):
        atr_sum += tr[i]
        if i >= 14:
            atr_sum -= tr[i-14]
        if i >= 13:  # 14 periods minimum
            atr[i] = atr_sum / 14.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(13, 20, 14)  # EMA13, Vol MA(20), ATR(14)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema13_6h[i]) or np.isnan(ema50_12h_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: bear power exceeds bull power OR stoploss
            if (bear_power[i] > bull_power[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: bull power exceeds bear power OR stoploss
            if (bull_power[i] > bear_power[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Trend filter: 12h EMA50 direction
            uptrend = close[i] > ema50_12h_aligned[i]
            downtrend = close[i] < ema50_12h_aligned[i]
            
            # Volume filter: participation confirmation
            vol_filter = volume[i] > (1.8 * vol_ma[i])
            
            # Enter long: bull power strong AND uptrend AND volume
            long_entry = (bull_power[i] > 0 and uptrend and vol_filter)
            # Enter short: bear power strong AND downtrend AND volume
            short_entry = (bear_power[i] > 0 and downtrend and vol_filter)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

---  END OF FILE ---