#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot and Volume Confirmation
Hypothesis: 6h Donchian breakouts aligned with weekly pivot levels and volume confirmation capture directional moves in both bull and bear markets. Weekly pivots provide structural support/resistance that holds across regimes.
Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_weekly_pivot_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for weekly pivots (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: P = (H+L+C)/3, S1 = 2P-H, R1 = 2P-L
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot = (high_1w + low_1w + close_1w) / 3
    s1 = 2 * pivot - high_1w
    r1 = 2 * pivot - low_1w
    
    # Align weekly pivots to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian channels (20 periods)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 6h ATR for stoploss
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
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: break below S1 (weekly support) OR stoploss
            if (close[i] <= s1_aligned[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: break above R1 (weekly resistance) OR stoploss
            if (close[i] >= r1_aligned[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume and pivot alignment
            # Long: break above Donchian high with volume, above weekly pivot
            long_breakout = (close[i] > donch_high[i] and
                           volume[i] > 1.5 * vol_ma[i] and
                           close[i] > pivot_aligned[i])
            # Short: break below Donchian low with volume, below weekly pivot
            short_breakout = (close[i] < donch_low[i] and
                            volume[i] > 1.5 * vol_ma[i] and
                            close[i] < pivot_aligned[i])
            
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
6h Camarilla Pivot Reversal with Volume Confirmation
Hypothesis: Price reverses at Camarilla levels (S3/R3) with volume confirmation in ranging markets, and breaks through S4/R4 with volume in trending markets. Works in both bull and bear regimes by adapting to market structure.
Target: 70-140 total trades over 4 years (18-35/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot points from previous day
    # Typical price = (H+L+C)/3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Range = H - L
    range_1d = df_1d['high'] - df_1d['low']
    # Camarilla levels
    s3 = typical_price - 1.1 * range_1d / 6
    s4 = typical_price - 2.0 * range_1d / 6
    r3 = typical_price + 1.1 * range_1d / 6
    r4 = typical_price + 2.0 * range_1d / 6
    
    # Align Camarilla levels to 6h
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h EMA(20) for trend filter
    ema_fast = pd.Series(close).ewm(span=20, adjust=False).mean().values
    
    # 6h volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 6h ATR for stoploss
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
    start = 20  # For EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_fast[i]) or np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: reach R3 (take profit) OR stoploss
            if (close[i] >= r3_aligned[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: reach S3 (take profit) OR stoploss
            if (close[i] <= s3_aligned[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Camarilla reversals/breakouts with volume
            # Long reversal at S3: price touches S3 with volume and closes above
            long_reversal = (low[i] <= s3_aligned[i] and
                           close[i] > s3_aligned[i] and
                           volume[i] > 1.5 * vol_ma[i])
            # Short reversal at R3: price touches R3 with volume and closes below
            short_reversal = (high[i] >= r3_aligned[i] and
                            close[i] < r3_aligned[i] and
                            volume[i] > 1.5 * vol_ma[i])
            # Long breakout above R4 with volume (trending market)
            long_breakout = (close[i] > r4_aligned[i] and
                           volume[i] > 2.0 * vol_ma[i] and
                           close[i] > ema_fast[i])
            # Short breakout below S4 with volume (trending market)
            short_breakout = (close[i] < s4_aligned[i] and
                            volume[i] > 2.0 * vol_ma[i] and
                            close[i] < ema_fast[i])
            
            if long_reversal or long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_reversal or short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Ichimoku Cloud with TK Cross and Volume Filter
Hypothesis: Ichimoku TK (Tenkan-Kijun) cross acts as momentum signal, with cloud (Senkou Span A/B) from 1d providing macro trend filter. Volume confirms institutional participation. Works in both bull (price above cloud) and bear (price below cloud) markets.
Target: 60-120 total trades over 4 years (15-30/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_tk_cross_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Ichimoku cloud (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = df_1d['high'].rolling(window=9, min_periods=9).max()
    low_9 = df_1d['low'].rolling(window=9, min_periods=9).min()
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = df_1d['high'].rolling(window=26, min_periods=26).max()
    low_26 = df_1d['low'].rolling(window=26, min_periods=26).min()
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = df_1d['high'].rolling(window=52, min_periods=52).max()
    low_52 = df_1d['low'].rolling(window=52, min_periods=52).min()
    senkou_b = ((high_52 + low_52) / 2).shift(26)
    
    # Align Ichimoku components to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b.values)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 6h ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (need 52+26=78 periods for Senkou B)
    start = 78
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud color and position
        # Green cloud: Senkou A > Senkou B (bullish)
        # Red cloud: Senkou A < Senkou B (bearish)
        # Cloud top: max(Senkou A, Senkou B)
        # Cloud bottom: min(Senkou A, Senkou B)
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Check exits
        if position == 1:  # long position
            # Exit: TK cross down OR price below cloud OR stoploss
            if (tenkan_aligned[i] < kijun_aligned[i] or
                close[i] < cloud_bottom or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: TK cross up OR price above cloud OR stoploss
            if (tenkan_aligned[i] > kijun_aligned[i] or
                close[i] > cloud_top or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: TK cross with volume and cloud filter
            # Long: TK cross up (Tenkan > Kijun) with volume, price above cloud
            long_signal = (tenkan_aligned[i] > kijun_aligned[i] and
                         volume[i] > 1.5 * vol_ma[i] and
                         close[i] > cloud_top)
            # Short: TK cross down (Tenkan < Kijun) with volume, price below cloud
            short_signal = (tenkan_aligned[i] < kijun_aligned[i] and
                          volume[i] > 1.5 * vol_ma[i] and
                          close[i] < cloud_bottom)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Elder Ray Index with Weekly Trend and Volume Filter
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) captures buying/selling pressure. Combined with weekly EMA trend filter and volume confirmation, it works in both bull (buy on bull power + weekly uptrend) and bear (sell on bear power + weekly downtrend) markets.
Target: 90-180 total trades over 4 years (23-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_weekly_trend_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for weekly trend (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA(40) for trend
    close_1w = df_1w['close'].values
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Buying pressure
    bear_power = low - ema_13   # Selling pressure
    
    # 6h volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 6h ATR for stoploss
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
    start = 20  # For volume MA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or
            np.isnan(ema_40_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine weekly trend
        weekly_uptrend = close[i] > ema_40_1w_aligned[i]
        weekly_downtrend = close[i] < ema_40_1w_aligned[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: bear power turns positive (selling pressure) OR stoploss
            if (bear_power[i] > 0 or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: bull power turns negative (buying pressure) OR stoploss
            if (bull_power[i] < 0 or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Elder Ray with weekly trend and volume
            # Long: bull power positive (buying pressure) with volume and weekly uptrend
            long_signal = (bull_power[i] > 0 and
                         volume[i] > 1.5 * vol_ma[i] and
                         weekly_uptrend)
            # Short: bear power negative (selling pressure) with volume and weekly downtrend
            short_signal = (bear_power[i] < 0 and
                          volume[i] > 1.5 * vol_ma[i] and
                          weekly_downtrend)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

}