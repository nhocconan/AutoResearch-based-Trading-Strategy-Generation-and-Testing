#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
Hypothesis: Combining 6h Donchian breakouts with weekly pivot direction (from 1d data) filters out false breakouts.
Weekly pivot provides long-term bias: only take long breaks above weekly pivot in uptrend,
short breaks below weekly pivot in downtrend. Volume confirmation ensures breakout strength.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_pivot_volume_v1"
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
    
    # 14-period ATR for stoploss
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
    
    # Get weekly pivot from daily data (using 1d as proxy for weekly pivot calculation)
    # We'll calculate weekly pivot points using daily OHLC from the past week
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) > 0:
        # Calculate weekly pivot points: (Prior Week High + Low + Close) / 3
        # We need to resample daily data to weekly to get prior week's OHLC
        # But since we can't resample, we'll approximate using rolling window
        # Alternative: use daily data to calculate pivot for each day, then align
        # Standard pivot: (High + Low + Close) / 3
        # For weekly pivot, we use prior week's values
        if len(df_1d) >= 5:  # Need at least 5 days for a week
            # Calculate rolling weekly high, low, close (using 5-day window as proxy for week)
            weekly_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1)  # Prior week
            weekly_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1)
            weekly_close = df_1d['close'].rolling(window=5, min_periods=5).mean().shift(1)
            
            # Weekly pivot point
            weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
            
            # Align to 6h timeframe
            weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot.values)
        else:
            weekly_pivot_aligned = np.full(n, np.nan)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 30  # For Donchian and weekly pivot
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(weekly_pivot_aligned[i]):
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
        volume_filter = volume[i] > vol_ma * 1.5  # Volume > 1.5x average
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR below weekly pivot
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < weekly_pivot_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR above weekly pivot
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > weekly_pivot_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Donchian breakout + volume + weekly pivot filter
            # Minimum holding period: only allow new entry after 20 bars flat
            if bars_since_entry >= 20:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Weekly pivot filter: only go long if above weekly pivot, short if below
                above_pivot = close[i] > weekly_pivot_aligned[i]
                below_pivot = close[i] < weekly_pivot_aligned[i]
                
                if bull_breakout and volume_filter and above_pivot:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_breakout and volume_filter and below_pivot:
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
6h Camarilla Pivot Reversal + Volume Spike + Weekly Trend Filter
Hypothesis: Camarilla levels (R3/S3, R4/S4) from daily data provide high-probability reversal zones.
Trades fade at R3/S3 with volume confirmation, breakout continuation at R4/S4.
Weekly trend filter (using 1d data) ensures trading in direction of higher timeframe trend.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_v1"
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
    
    # 14-period ATR for stoploss
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
    
    # Get Camarilla levels from daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) > 0:
        # Camarilla levels use previous day's OHLC
        # Typical formula based on previous day's range
        prev_high = df_1d['high'].shift(1)
        prev_low = df_1d['low'].shift(1)
        prev_close = df_1d['close'].shift(1)
        
        # Calculate pivot and ranges
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_val = prev_high - prev_low
        
        # Camarilla levels
        r4 = pivot + (range_val * 1.1 / 2)
        r3 = pivot + (range_val * 1.1 / 4)
        s3 = pivot - (range_val * 1.1 / 4)
        s4 = pivot - (range_val * 1.1 / 2)
        
        # Align to 6h timeframe
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    else:
        r4_aligned = r3_aligned = s3_aligned = s4_aligned = np.full(n, np.nan)
    
    # Weekly trend filter from daily data (using 5-day EMA as proxy for weekly)
    if len(df_1d) > 0:
        ema_5d = df_1d['close'].ewm(span=5, min_periods=5).mean()
        ema_aligned = align_htf_to_ltf(prices, df_1d, ema_5d.values)
        weekly_uptrend = close > ema_aligned  # Price above 5-day EMA = uptrend
        weekly_downtrend = close < ema_aligned  # Price below 5-day EMA = downtrend
    else:
        weekly_uptrend = weekly_downtrend = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # Need at least 1 day of data
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or \
           np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.5  # Volume > 1.5x average
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price crosses S3 (mean reversion target) or stoploss
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < s3_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price crosses R3 (mean reversion target) or stoploss
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > r3_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Camarilla reversals + volume + weekly trend filter
            # Minimum holding period: only allow new entry after 15 bars flat
            if bars_since_entry >= 15:
                # Fade at R3/S3: sell at R3, buy at S3
                # But only with volume confirmation and in direction of weekly trend
                near_r3 = abs(high[i] - r3_aligned[i]) < (r3_aligned[i] * 0.001)  # Within 0.1% of R3
                near_s3 = abs(low[i] - s3_aligned[i]) < (s3_aligned[i] * 0.001)  # Within 0.1% of S3
                
                # Breakout at R4/S4: buy above R4, sell below S4
                breakout_r4 = close[i] > r4_aligned[i]
                breakout_s4 = close[i] < s4_aligned[i]
                
                if near_r3 and volume_filter and weekly_downtrend:
                    # Fade at R3 in downtrend: short
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif near_s3 and volume_filter and weekly_uptrend:
                    # Fade at S3 in uptrend: long
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif breakout_r4 and volume_filter and weekly_uptrend:
                    # Breakout above R4 in uptrend: long
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif breakout_s4 and volume_filter and weekly_downtrend:
                    # Breakout below S4 in downtrend: short
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
6h Ichimoku Cloud + Daily Trend Filter + Volume Confirmation
Hypothesis: Ichimoku provides comprehensive trend, support/resistance, and momentum signals.
Tenkan/Kijun cross signals entry, cloud (Senkou Span A/B) acts as dynamic support/resistance.
Daily trend filter ensures alignment with higher timeframe. Volume confirms signal strength.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_daily_trend_v1"
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
    
    # 14-period ATR for stoploss
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
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = np.full(n, np.nan)
    period9_low = np.full(n, np.nan)
    if n >= 9:
        for i in range(8, n):
            period9_high[i] = np.max(high[i-8:i+1])
            period9_low[i] = np.min(low[i-8:i+1])
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = np.full(n, np.nan)
    period26_low = np.full(n, np.nan)
    if n >= 26:
        for i in range(25, n):
            period26_high[i] = np.max(high[i-25:i+1])
            period26_low[i] = np.min(low[i-25:i+1])
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = np.full(n, np.nan)
    period52_low = np.full(n, np.nan)
    if n >= 52:
        for i in range(51, n):
            period52_high[i] = np.max(high[i-51:i+1])
            period52_low[i] = np.min(low[i-51:i+1])
    senkou_b = (period52_high + period52_low) / 2.0
    
    # Get daily trend filter (using 50-period EMA as proxy for weekly/daily trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) > 0:
        # Calculate 50-period EMA on daily data
        ema_50 = df_1d['close'].ewm(span=50, min_periods=50).mean()
        ema_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
        # Daily trend: price above/below 50 EMA
        daily_uptrend = close > ema_aligned
        daily_downtrend = close < ema_aligned
    else:
        daily_uptrend = daily_downtrend = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 52  # Need Senkou Span B
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or \
           np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.5  # Volume > 1.5x average
        
        # Determine cloud direction (green = bullish, red = bearish)
        # Cloud is bullish when Senkou A > Senkou B
        cloud_bullish = senkou_a[i] > senkou_b[i]
        cloud_bearish = senkou_a[i] < senkou_b[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below cloud OR Tenkan/Kijun cross down
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < senkou_b[i] or  # Below cloud (support)
                tenkan[i] < kijun[i] or    # Tenkan/Kijun death cross
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above cloud OR Tenkan/Kijun cross up
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > senkou_a[i] or  # Above cloud (resistance)
                tenkan[i] > kijun[i] or    # Tenkan/Kijun golden cross
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Tenkan/Kijun cross + cloud filter + daily trend + volume
            # Minimum holding period: only allow new entry after 20 bars flat
            if bars_since_entry >= 20:
                # Tenkan/Kijun cross
                tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
                tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
                
                if tk_cross_up and cloud_bullish and daily_uptrend and volume_filter:
                    # Golden cross in bullish cloud with uptrend: long
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif tk_cross_down and cloud_bearish and daily_downtrend and volume_filter:
                    # Death cross in bearish cloud with downtrend: short
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
6h Elder Ray Power + Volume Spike + Weekly Trend Filter
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures buying/selling pressure.
Combined with 13-period EMA trend and volume spikes, it captures strong directional moves.
Weekly trend filter (using 1d data) ensures trading in direction of higher timeframe.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_v1"
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
    
    # 14-period ATR for stoploss
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
    
    # 13-period EMA for Elder Ray and trend
    ema13 = np.full(n, np.nan)
    if n >= 13:
        ema13[12] = np.mean(close[:13])
        for i in range(13, n):
            ema13[i] = (close[i] * 2 / (13 + 1)) + (ema13[i-1] * (11 / (13 + 1)))
    
    # Elder Ray Power
    bull_power = high - ema13  # Buying strength
    bear_power = ema13 - low   # Selling strength
    
    # Get weekly trend filter from daily data (using 20-period EMA as proxy for weekly)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) > 0:
        # Calculate 20-period EMA on daily data
        ema_20d = df_1d['close'].ewm(span=20, min_periods=20).mean()
        ema_aligned = align_htf_to_ltf(prices, df_1d, ema_20d.values)
        # Weekly trend: price above/below 20-day EMA
        weekly_uptrend = close > ema_aligned
        weekly_downtrend = close < ema_aligned
    else:
        weekly_uptrend = weekly_downtrend = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # Need EMA13 and enough data
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.5  # Volume > 1.5x average
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Bear power expands (selling pressure increases) OR stoploss
            # Stoploss: price drops 2*ATR below entry
            if (bear_power[i] > bear_power[i-1] * 1.5 or  # Bear power expanding
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: Bull power expands (buying pressure increases) OR stoploss
            # Stoploss: price rises 2*ATR above entry
            if (bull_power[i] > bull_power[i-1] * 1.5 or  # Bull power expanding
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Elder Ray divergence + volume + weekly trend filter
            # Minimum holding period: only allow new entry after 20 bars flat
            if bars_since_entry >= 20:
                # Bullish: Bull power increasing AND above EMA13
                bull_power_rising = bull_power[i] > bull_power[i-1]
                price_above_ema = close[i] > ema13[i]
                
                # Bearish: Bear power increasing AND below EMA13
                bear_power_rising = bear_power[i] > bear_power[i-1]
                price_below_ema = close[i] < ema13[i]
                
                if bull_power_rising and price_above_ema and volume_filter and weekly_uptrend:
                    # Strong buying pressure with uptrend: long
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif bear_power_rising and price_below_ema and volume_filter and weekly_downtrend:
                    # Strong selling pressure with downtrend: short
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
6h ADX + Williams Alligator + Volume Confirmation
Hypothesis: ADX measures trend strength, Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and entry points.
Combining ADX > 25 (strong trend) with Alligator alignment (Lips > Teeth > Jaw for uptrend) provides high-probability trades.
Volume confirms breakout strength. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_alligator_v1"
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
    
    # 14-period ATR for stoploss
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
    
    # Williams Alligator (13, 8, 5 period SMAs with future shifts)
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    # SMMA = Smoothed Moving Average (similar to Wilder's smoothing)
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            # First value is simple average
            result[period-1] = np.mean(arr[:period])
            # Subsequent values: (prev * (period-1) + current) / period
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate SMMA then apply shifts
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if n > 8:
        jaw[8:] = jaw_raw[:-8]
    if n > 5:
        teeth[5:] = teeth_raw[:-5]
    if n > 3:
        lips[3:] = lips_raw[:-3]
    
    # 14-period ADX
    adx = np.full(n, np.nan)
    if n >= 14:
        # +DM and -DM
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # True Range (same as ATR calculation)
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        
        # Smoothed values
        tr14 = np.full(n, np.nan)
        plus_dm14 = np.full(n, np.nan)
        minus_dm14 = np.full(n, np.nan)
        
        if len(tr) >= 14:
            tr14[14] = np.sum(tr[:14])
            plus_dm14[14] = np.sum(plus_dm[:14])
            minus_dm14[14] = np.sum(minus_dm[:14])
            
            for i in range(15, n):
                tr14[i] = tr14[i-1] - (tr14[i-1] / 14) + tr[i-1]
                plus_dm14[i] = plus_dm14[i-1] - (plus_dm14[i