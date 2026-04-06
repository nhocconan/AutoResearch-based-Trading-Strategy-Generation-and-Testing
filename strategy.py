#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 12h EMA Trend + Volume Spike + ATR Stop
Hypothesis: Combines price channel breakouts with higher timeframe trend bias
and volume confirmation to capture momentum while avoiding chop.
Works in bull (breakouts with trend) and bear (short breakdowns with trend).
Designed for moderate trade frequency (~15-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_12htrend_vol_v1"
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
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # 12h EMA20 for trend bias
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 20:
        ema_12h[19] = np.mean(close_12h[:20])
        for i in range(20, len(close_12h)):
            ema_12h[i] = (close_12h[i] * 2 + ema_12h[i-1] * 18) / 20
    
    # Trend bias: above EMA = bullish, below = bearish
    trend_bias_12h = np.where(close_12h > ema_12h, 1, -1)
    
    # Align to 6h timeframe
    trend_bias_aligned = align_htf_to_ltf(prices, df_12h, trend_bias_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(trend_bias_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 2.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR against 12h trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                trend_bias_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR against 12h trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                trend_bias_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Donchian breakout + 12h trend + volume spike
            # Minimum holding period: only allow new entry after 15 bars flat
            if bars_since_entry >= 15:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Long: bullish breakout with bullish 12h trend and volume
                if bull_breakout and trend_bias_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: bearish breakout with bearish 12h trend and volume
                elif bear_breakout and trend_bias_aligned[i] == -1 and volume_filter:
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
6h Camarilla Pivot Reversal with Volume Confirmation
Hypothesis: Camarilla pivot levels (based on 1d OHLC) act as strong support/resistance.
Price tends to reverse at R3/S3 and break through R4/S4 with volume.
Works in ranging markets (reversions at R3/S3) and trending markets (breakouts at R4/S4).
Volume filter ensures only significant moves are traded.
Designed for low trade frequency (~10-25/year) to minimize fee drag in 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_rev_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Get 1d OHLC for Camarilla calculation (use previous day's values)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC (shifted by 1 to avoid look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for each 1d bar
    # R4 = Close + (High - Low) * 1.1/2
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    # S4 = Close - (High - Low) * 1.1/2
    r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    s4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For volume filter
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches S3 (target) or stoploss hit
            if (close[i] <= s3_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price reaches R3 (target) or stoploss hit
            if (close[i] >= r3_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Camarilla reversals/breakouts with volume
            # Minimum holding period: only allow new entry after 10 bars flat
            if bars_since_entry >= 10:
                # Reversal at S3 (long) or R3 (short)
                near_s3 = abs(close[i] - s3_aligned[i]) < (atr[i] * 0.5)
                near_r3 = abs(close[i] - r3_aligned[i]) < (atr[i] * 0.5)
                
                # Breakout above R4 or below S4
                breakout_up = close[i] > r4_aligned[i]
                breakout_down = close[i] < s4_aligned[i]
                
                # Long: reversal at S3 with volume OR breakout above R4 with volume
                if (near_s3 and volume_filter) or (breakout_up and volume_filter):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: reversal at R3 with volume OR breakout below S4 with volume
                elif (near_r3 and volume_filter) or (breakout_down and volume_filter):
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
6h Ichimoku Cloud with 1d Trend Filter and Volume Confirmation
Hypothesis: Ichimoku provides comprehensive trend, momentum, and support/resistance.
TK cross (Tenkan/Kijun) signals momentum shifts, cloud acts as dynamic S/R.
Using 1d trend filter ensures alignment with higher timeframe direction.
Volume confirms signal strength. Works in all markets by adapting to trend.
Designed for moderate trade frequency (~15-35/year) to balance opportunity and cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1dtrend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = np.full(n, np.nan)
    period9_low = np.full(n, np.nan)
    for i in range(n):
        if i >= 8:
            period9_high[i] = np.max(high[i-8:i+1])
            period9_low[i] = np.min(low[i-8:i+1])
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = np.full(n, np.nan)
    period26_low = np.full(n, np.nan)
    for i in range(n):
        if i >= 25:
            period26_high[i] = np.max(high[i-25:i+1])
            period26_low[i] = np.min(low[i-25:i+1])
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = np.full(n, np.nan)
    period52_low = np.full(n, np.nan)
    for i in range(n):
        if i >= 51:
            period52_high[i] = np.max(high[i-51:i+1])
            period52_low[i] = np.min(low[i-51:i+1])
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Get 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 48) / 50
    
    # 1d trend: above EMA50 = bullish, below = bearish
    trend_bias_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_bias_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period (need 52 periods for Senkou B)
    start = 52
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or np.isnan(trend_bias_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.8
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below cloud OR TK cross turns bearish
            # Stoploss: price drops 2*ATR below entry
            cloud_bottom = min(senkou_a[i], senkou_b[i])
            if (close[i] < cloud_bottom or
                (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]) or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above cloud OR TK cross turns bullish
            # Stoploss: price rises 2*ATR above entry
            cloud_top = max(senkou_a[i], senkou_b[i])
            if (close[i] > cloud_top or
                (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]) or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: TK cross + cloud filter + 1d trend + volume
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # TK cross: Tenkan crosses Kijun
                tk_cross_bull = (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1])
                tk_cross_bear = (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1])
                
                # Cloud relationship: price above/below cloud
                cloud_top = max(senkou_a[i], senkou_b[i])
                cloud_bottom = min(senkou_a[i], senkou_b[i])
                price_above_cloud = close[i] > cloud_top
                price_below_cloud = close[i] < cloud_bottom
                
                # Long: bullish TK cross above cloud with bullish 1d trend and volume
                if (tk_cross_bull and price_above_cloud and
                    trend_bias_aligned[i] == 1 and volume_filter):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: bearish TK cross below cloud with bearish 1d trend and volume
                elif (tk_cross_bear and price_below_cloud and
                      trend_bias_aligned[i] == -1 and volume_filter):
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
6h Elder Ray Power with ADX Regime Filter and Volume Confirmation
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) 
measures bull/bear strength relative to trend. Combines with ADX to distinguish
trending vs ranging markets. In trending markets (ADX>25), follow Elder Ray polarity.
In ranging markets (ADX<20), fade extreme Elder Ray readings. Volume confirms.
Adaptive approach works in all market regimes. Designed for low-moderate frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elderray_adx_regime_v1"
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
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # EMA13 for Elder Ray
    ema13 = np.full(n, np.nan)
    if n >= 13:
        ema13[12] = np.mean(close[:13])
        for i in range(13, n):
            ema13[i] = (close[i] * 2 + ema13[i-1] * 11) / 12
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # ADX calculation (14-period)
    # +DM, -DM, TR
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        elif low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Smoothed values
    atr_14 = np.full(n, np.nan)
    plus_dm_14 = np.full(n, np.nan)
    minus_dm_14 = np.full(n, np.nan)
    
    if n >= 14:
        # Initial averages
        atr_14[13] = np.mean(tr[1:14])
        plus_dm_14[13] = np.sum(plus_dm[1:14])
        minus_dm_14[13] = np.sum(minus_dm[1:14])
        
        # Wilder's smoothing
        for i in range(14, n):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
            plus_dm_14[i] = (plus_dm_14[i-1] * 13 + plus_dm[i]) / 14
            minus_dm_14[i] = (minus_dm_14[i-1] * 13 + minus_dm[i]) / 14
    
    # ADX = 100 * smoothed(|+DI - -DI|) / (+DI + -DI)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    adx = np.full(n, np.nan)
    
    if n >= 14:
        for i in range(14, n):
            if atr_14[i] != 0:
                plus_di[i] = 100 * plus_dm_14[i] / atr_14[i]
                minus_di[i] = 100 * minus_dm_14[i] / atr_14[i]
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # Smooth DX to get ADX
        if n >= 27:
            adx[26] = np.mean(dx[14:28])
            for i in range(27, n):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 30  # For ADX
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.6
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Elder Ray turns negative OR stoploss hit
            if (bull_power[i] < 0 or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: Elder Ray turns positive OR stoploss hit
            if (bear_power[i] < 0 or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries based on regime
            # Minimum holding period: only allow new entry after 15 bars flat
            if bars_since_entry >= 15:
                # Regime classification
                is_trending = adx[i] > 25
                is_ranging = adx[i] < 20
                
                if is_trending:
                    # Trending market: follow Elder Ray polarity
                    strong_bull = bull_power[i] > (atr[i] * 0.8)
                    strong_bear = bear_power[i] > (atr[i] * 0.8)
                    
                    if strong_bull and volume_filter:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                        bars_since_entry = 0
                    elif strong_bear and volume_filter:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
                        bars_since_entry = 0
                    else:
                        signals[i] = 0.0
                        bars_since_entry += 1
                elif is_ranging:
                    # Ranging market: fade extreme Elder Ray readings
                    # Normalize Elder Ray by ATR for comparison
                    bull_norm = bull_power[i] / atr[i] if atr[i] > 0 else 0
                    bear_norm = bear_power[i] / atr[i] if atr[i] > 0 else 0
                    
                    # Extreme readings: >1.5 ATR away from EMA
                    extreme_bull = bull_norm > 1.5
                    extreme_bear = bear_norm > 1.5
                    
                    if extreme_bear and volume_filter:
                        # Fade bullish extreme - go short
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
                        bars_since_entry = 0
                    elif extreme_bull and volume_filter:
                        # Fade bearish extreme - go long
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                        bars_since_entry = 0
                    else:
                        signals[i] = 0.0
                        bars_since_entry += 1
                else:
                    # Transition regime: no trade
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals

=======
#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 12h EMA Trend + Volume Spike + ATR Stop
Hypothesis: Combines price channel breakouts with higher timeframe trend bias
and volume confirmation to capture momentum while avoiding chop.
Works in bull (breakouts with trend) and bear (short breakdowns with trend).
Designed for moderate trade frequency (~15-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_12htrend_vol_v1"
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
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # 12h EMA20 for trend bias
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 20:
        ema_12h[19] = np.mean(close_12h[:20])
        for i in range(20, len(close_12h)):
            ema_12h[i] = (close_12h[i] * 2 + ema_12h[i-1] * 18) / 20
    
    # Trend bias: above EMA = bullish, below = bearish
    trend_bias_12h = np.where(close_12h > ema_12h, 1, -1)
    
    # Align to 6h timeframe
    trend_bias_aligned = align_htf_to_ltf(prices, df_12h, trend_bias_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(trend_bias_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol