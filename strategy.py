#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d EMA trend filter and volume confirmation.
# Uses 6h price channel breakouts aligned with 1d momentum to capture trending moves.
# Volume confirmation ensures institutional participation. Works in bull markets (breakouts above upper band)
# and bear markets (breakdowns below lower band). Target: 75-200 total trades over 4 years (19-50/year).

name = "exp_13467_6h_donchian20_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_PERIOD, DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: price above/below 1d EMA
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Breakout signals using Donchian channels
        breakout_up = volume_ok and uptrend and (high[i] > highest_high[i-1])
        breakout_down = volume_ok and downtrend and (low[i] < lowest_low[i-1])
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels with weekly trend filter and volume confirmation.
# Uses daily Camarilla levels (H3/L3 for mean reversion, H4/L4 for breakout) aligned with weekly trend.
# Volume confirmation filters noise. Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13467_6h_camarilla_pivot_weekly_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
WEEKLY_EMA_PERIOD = 21

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_weekly = df_weekly['close'].values
    ema_weekly = calculate_ema(close_weekly, WEEKLY_EMA_PERIOD)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla levels
    # Use previous day's OHLC for today's levels
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    
    # First value: use first available (no lookback)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    pivot = (high_shift + low_shift + close_shift) / 3.0
    range_val = high_shift - low_shift
    
    # Camarilla levels
    h3 = pivot + (range_val * 1.1 / 6.0)
    l3 = pivot - (range_val * 1.1 / 6.0)
    h4 = pivot + (range_val * 1.1 / 2.0)
    l4 = pivot - (range_val * 1.1 / 2.0)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WEEKLY_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(ema_weekly_aligned[i]) or np.isnan(h3[i]) or np.isnan(l3[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Weekly trend filter
        uptrend = close[i] > ema_weekly_aligned[i]
        downtrend = close[i] < ema_weekly_aligned[i]
        
        # Camarilla signals
        # Mean reversion at H3/L3 in ranging markets
        # Breakout continuation at H4/L4 in trending markets
        if position == 0:
            # Long signals
            if volume_ok and uptrend and low[i] <= l3 and close[i] > l3:
                # Bounce off L3 in uptrend
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif volume_ok and downtrend and high[i] >= h3 and close[i] < h3:
                # Rejection at H3 in downtrend
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif volume_ok and uptrend and high[i] >= h4:
                # Breakout above H4 in uptrend
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif volume_ok and downtrend and low[i] <= l4:
                # Breakdown below L4 in downtrend
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with daily trend filter and volume confirmation.
# Uses Ichimoku (TK cross + price above/below cloud) aligned with daily EMA trend.
# Volume confirmation ensures institutional participation. Works in all market regimes.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13467_6h_ichimoku_daily_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
DAILY_EMA_PERIOD = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend filter
    close_daily = df_daily['close'].values
    ema_daily = calculate_ema(close_daily, DAILY_EMA_PERIOD)
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    highest_high_9 = pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max().values
    lowest_low_9 = pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min().values
    tenkan = (highest_high_9 + lowest_low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    highest_high_26 = pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max().values
    lowest_low_26 = pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min().values
    kijun = (highest_high_26 + lowest_low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    highest_high_52 = pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max().values
    lowest_low_52 = pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min().values
    senkou_b = (highest_high_52 + lowest_low_52) / 2.0
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(SENKOU_B_PERIOD, DAILY_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily EMA not available
        if np.isnan(ema_daily_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Daily trend filter
        uptrend = close[i] > ema_daily_aligned[i]
        downtrend = close[i] < ema_daily_aligned[i]
        
        # Ichimoku signals
        # Kumo (cloud) top and bottom
        # Note: Senkou spans are plotted ahead, but we use current values for cloud thickness
        # For simplicity, we use current Senkou A/B as cloud boundaries
        if senkou_a[i] >= senkou_b[i]:
            cloud_top = senkou_a[i]
            cloud_bottom = senkou_b[i]
        else:
            cloud_top = senkou_b[i]
            cloud_bottom = senkou_a[i]
        
        # TK cross signals with cloud filter
        if position == 0:
            # Long: TK cross bullish + price above cloud + uptrend
            if tenkan[i-1] <= kijun[i-1] and tenkan[i] > kijun[i] and close[i] > cloud_top and uptrend and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: TK cross bearish + price below cloud + downtrend
            elif tenkan[i-1] >= kijun[i-1] and tenkan[i] < kijun[i] and close[i] < cloud_bottom and downtrend and volume_ok:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long if price drops below cloud bottom or TK cross bearish
            if close[i] < cloud_bottom or (tenkan[i-1] > kijun[i-1] and tenkan[i] < kijun[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short if price rises above cloud top or TK cross bullish
            if close[i] > cloud_top or (tenkan[i-1] < kijun[i-1] and tenkan[i] > kijun[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with Elder Ray power and weekly trend filter.
# Uses Alligator (jaw/teeth/lips) to identify trends, Elder Ray to measure bull/bear power.
# Weekly EMA filter ensures alignment with higher timeframe trend.
# Volume confirmation reduces false signals. Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13467_6h_alligator_elder_ray_weekly_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13
ALLIGATOR_TEETH_PERIOD = 8
ALLIGATOR_LIPS_PERIOD = 5
ELDER_RAY_PERIOD = 13
WEEKLY_EMA_PERIOD = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_weekly = df_weekly['close'].values
    ema_weekly = calculate_ema(close_weekly, WEEKLY_EMA_PERIOD)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator
    jaw = calculate_ema(high, ALLIGATOR_JAW_PERIOD)  # Jaw (blue) - 13-period SMMA smoothed
    teeth = calculate_ema(low, ALLIGATOR_TEETH_PERIOD)  # Teeth (red) - 8-period SMMA smoothed
    lips = calculate_ema(close, ALLIGATOR_LIPS_PERIOD)  # Lips (green) - 5-period SMMA smoothed
    
    # Elder Ray Power
    # Bull Power = High - EMA(close)
    # Bear Power = Low - EMA(close)
    ema_close = calculate_ema(close, ELDER_RAY_PERIOD)
    bull_power = high - ema_close
    bear_power = low - ema_close
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_JAW_PERIOD, WEEKLY_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(ema_weekly_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Weekly trend filter
        uptrend = close[i] > ema_weekly_aligned[i]
        downtrend = close[i] < ema_weekly_aligned[i]
        
        # Alligator signals: lips above teeth above jaw = uptrend, reverse = downtrend
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray confirmation: strong bull/bear power
        strong_bull = bull_power[i] > 0 and bull_power[i] > np.mean(bull_power[max(0, i-20):i+1])
        strong_bear = bear_power[i] < 0 and bear_power[i] < np.mean(bear_power[max(0, i-20):i+1])
        
        # Generate signals
        if position == 0:
            # Long: Alligator aligned up + Elder Ray bullish + weekly uptrend + volume
            if alligator_long and strong_bull and uptrend and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: Alligator aligned down + Elder Ray bearish + weekly downtrend + volume
            elif alligator_short and strong_bear and downtrend and volume_ok:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long if Alligator reverses or Elder Ray turns bearish
            if not alligator_long or not strong_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short if Alligator reverses or Elder Ray turns bullish
            if not alligator_short or not strong_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with ADX trend strength filter and volume confirmation.
# Uses Donchian channels for breakouts, ADX to filter only strong trends.
# Volume confirmation ensures institutional participation. Works in all market regimes.
# Target: 75-200 total trades over 4 years (19-50/year).

name = "exp_13467_6h_donchian20_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
ADX_PERIOD = 14
ADX_THRESHOLD = 25
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_dmi(high, low, close, period):
    """Calculate DMI components (+DI, -DI, ADX)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth using Wilder's smoothing (EMA with alpha=1/period)
    def WilderSmoothing(data, period):
        return pd.Series(data).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    tr_smoothed = WilderSmoothing(tr, period)
    dm_plus_smoothed = WilderSmoothing(dm_plus, period)
    dm_minus_smoothed = WilderSmoothing(dm_minus, period)
    
    # DI components
    plus_di = 100 * dm_plus_smoothed / tr_smoothed
    minus_di = 100 * dm_minus_smoothed / tr_smoothed
    
    # DX and ADX
    dx = np.where(tr_smoothed != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = WilderSmoothing(dx, period)
    
    return adx

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # ADX for trend strength
    adx = calculate_dmi(high, low, close, ADX_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, ADX_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(adx[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend strength filter
        strong_trend = adx[i] > ADX_THRESHOLD
        
        # Breakout signals using Donchian channels
        breakout_up = volume_ok and strong_trend and (high[i] > highest_high[i-1])
        breakout_down = volume_ok and strong_trend and (low[i] < lowest_low[i-1])
        
        # Generate signals
        if