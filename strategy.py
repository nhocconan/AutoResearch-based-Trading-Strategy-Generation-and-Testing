#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot levels from daily data with volume confirmation.
# Camarilla pivot levels (H4/L4) act as strong support/resistance in ranging markets.
# When price breaks above H4 with volume, it indicates bullish breakout; below L4 with volume indicates bearish breakout.
# In trending markets, price often respects H3/L3 levels for mean reversion.
# The strategy uses daily Camarilla levels for breakout/mean reversion signals with volume filter.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "exp_13287_6h_camarilla_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Daily Camarilla from previous day
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
USE_MEAN_REVERSION = True  # Also trade mean reversion at H3/L3

def calculate_pivot_points(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3
    range_val = high - low
    h4 = pivot + (range_val * 1.1 / 2)
    l4 = pivot - (range_val * 1.1 / 2)
    h3 = pivot + (range_val * 1.1 / 4)
    l3 = pivot - (range_val * 1.1 / 4)
    h2 = pivot + (range_val * 1.1 / 6)
    l2 = pivot - (range_val * 1.1 / 6)
    h1 = pivot + (range_val * 1.1 / 12)
    l1 = pivot - (range_val * 1.1 / 12)
    return h4, l4, h3, l3, h2, l2, h1, l1, pivot

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
    
    # Load daily data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    h4_1d, l4_1d, h3_1d, l3_1d, _, _, _, _, _ = calculate_pivot_points(high_1d, low_1d, close_1d)
    
    # Align to 6h timeframe
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(20, VOLUME_MA_PERIOD, ATR_PERIOD) + 1  # Need at least 1 day of data
    
    for i in range(start, n):
        # Skip if Camarilla levels not available
        if np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]):
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
        
        # Breakout signals: break above H4 or below L4 with volume
        breakout_up = volume_ok and (high[i] > h4_1d_aligned[i-1])
        breakout_down = volume_ok and (low[i] < l4_1d_aligned[i-1])
        
        # Mean reversion signals: reverse at H3/L3 (only if not in breakout mode)
        mean_rev_up = volume_ok and (low[i] <= l3_1d_aligned[i-1]) and (close[i] > l3_1d_aligned[i-1])
        mean_rev_down = volume_ok and (high[i] >= h3_1d_aligned[i-1]) and (close[i] < h3_1d_aligned[i-1])
        
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
            elif USE_MEAN_REVERSION and mean_rev_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif USE_MEAN_REVERSION and mean_rev_down:
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

# Hypothesis: 6-hour Elder Ray Index (Bull Power/Bear Power) with 12-hour EMA trend filter and volume confirmation.
# Elder Ray measures bull/bear power relative to EMA: Bull Power = High - EMA, Bear Power = Low - EMA.
# In bull markets, sustained Bull Power > 0 indicates strength; in bear markets, sustained Bear Power < 0 indicates weakness.
# The 12h EMA acts as trend filter to avoid counter-trend trades. Volume confirms genuine momentum.
# Works in both bull (buy strength) and bear (sell weakness) markets.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "exp_13287_elder_ray_6h_12h_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_RAY_PERIOD = 13  # EMA period for Elder Ray
TREND_EMA_PERIOD = 20  # 12h EMA for trend filter
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
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, TREND_EMA_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components: EMA and Bull/Bear Power
    ema_6h = calculate_ema(close, ELDER_RAY_PERIOD)
    bull_power = high - ema_6h  # Bull Power: High - EMA
    bear_power = low - ema_6h   # Bear Power: Low - EMA
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ELDER_RAY_PERIOD, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if EMA not available
        if np.isnan(ema_12h_aligned[i]):
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
        
        # Trend filter: 12h EMA slope (using close vs EMA)
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        # Elder Ray signals with volume and trend confirmation
        long_signal = volume_ok and uptrend and (bull_power[i] > 0) and (bull_power[i] > bull_power[i-1])
        short_signal = volume_ok and downtrend and (bear_power[i] < 0) and (bear_power[i] < bear_power[i-1])
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
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

# Hypothesis: 6-hour ADX combined with Williams Alligator (Jaw/Teeth/Lips) for trend strength and direction.
# ADX > 25 indicates strong trend (avoid ranging markets). Alligator lines show direction:
# Lips (5-period SMMA) above Teeth (8-period SMMA) above Jaw (13-period SMMA) = uptrend.
# Reverse for downtrend. This combo filters weak trends and whipsaws.
# Works in bull (catch strong uptrends) and bear (catch strong downtrends) markets.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "exp_13287_adx_alligator_6h_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ADX_THRESHOLD = 25
ALLIGATOR_JAW = 13   # SMMA period
ALLIGATOR_TEETH = 8   # SMMA period
ALLIGATOR_LIPS = 5    # SMMA period
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def smma(close, period):
    """Smoothed Moving Average (SMMA) - similar to Wilder's smoothing"""
    return pd.Series(close).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up = high - np.roll(high, 1)
    down = np.roll(low, 1) - low
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
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
    
    # ADX for trend strength
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # Williams Alligator (SMMA lines)
    jaw = smma(close, ALLIGATOR_JAW)
    teeth = smma(close, ALLIGATOR_TEETH)
    lips = smma(close, ALLIGATOR_LIPS)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD, ALLIGATOR_JAW, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not ready
        if np.isnan(adx[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
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
        
        # Alligator trend direction
        # Lips > Teeth > Jaw = uptrend
        # Lips < Teeth < Jaw = downtrend
        uptrend = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        downtrend = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # ADX trend strength filter
        strong_trend = adx[i] > ADX_THRESHOLD
        
        # Generate signals: only trade when strong trend + Alligator alignment + volume
        if position == 0:
            if strong_trend and uptrend and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif strong_trend and downtrend and volume_ok:
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

# Hypothesis: 6-hour Ichimoku Cloud with daily timeframe alignment for trend and momentum.
# Uses Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (26, 52 displacement), Chikou Span (26).
# Price above cloud = bullish bias, below cloud = bearish bias.
# TK cross (Tenkan-sen crossing Kijun-sen) provides entry signals in direction of cloud.
# Daily cloud filter ensures alignment with higher timeframe trend.
# Works in bull (buy above cloud) and bear (sell below cloud) markets.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "exp_13287_ichimoku_6h_1d_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_SPAN_B_PERIOD = 52
KUMO_SHIFT = 26
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (HH + LL) / 2 for 9 periods
    tenkan_sen = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                  pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (HH + LL) / 2 for 26 periods
    kijun_sen = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                 pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(KUMO_SHIFT)
    
    # Senkou Span B (Leading Span B): (HH + LL) / 2 for 52 periods shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).max() + 
                      pd.Series(low).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).min()) / 2).shift(KUMO_SHIFT)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods back
    chikou_span = pd.Series(close).shift(-KUMO_SHIFT)
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values, chikou_span.values

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
    
    # Load daily data ONCE before loop for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Ichimoku for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Determine daily cloud trend: price above/below cloud
    # Cloud top = max(senkou_a, senkou_b), cloud bottom = min(senkou_a, senkou_b)
    cloud_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    cloud_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    price_above_cloud_1d = close_1d > cloud_top_1d
    price_below_cloud_1d = close_1d < cloud_bottom_1d
    
    # Align to 6h timeframe
    price_above_cloud_1d_aligned = align_htf_to_ltf(prices, df_1d, price_above_cloud_1d.astype(float))
    price_below_cloud_1d_aligned = align_htf_to_ltf(prices, df_1d, price_below_cloud_1d.astype(float))
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Ichimoku for entry signals
    tenkan_6h, kijun_6h, _, _, _ = calculate_ichimoku(high, low, close)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_SPAN_B_PERIOD, KUMO_SHIFT, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if daily Ichimoku not available
        if np.isnan(price_above_cloud_1d_aligned[i]) or np.isnan(price_below_cloud_1d_aligned[i]):
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
        
        # 6h TK cross signals
        tk_cross_up = (tenkan_6h[i-1] <= kijun_6h[i-1]) and (tenkan_6h[i] > kijun_6h[i])
        tk_cross_down = (tenkan_6h[i-1] >= kijun_6h[i-1]) and (tenkan_6h[i] < kijun_6h[i])
        
        # Generate signals: trade TK cross in direction of daily cloud
        if position == 0:
            if price_above_cloud_1d_aligned[i-1] and tk_cross_up and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif price_below_cloud_1d_aligned[i-1] and tk_cross_down and volume_ok:
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

# Hypothesis: 6-hour Connors RSI (CRSI) with 1-week trend filter and volume confirmation.
# CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
# Oversold: CRSI < 10, Overbought: CRSI > 90
# Weekly EMA trend filter ensures trading in direction of higher timeframe momentum.
# Volume confirms genuine momentum. Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend).
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "exp_13287_crsi_6h_1w_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CRSI_RSI_PERIOD = 3
CRSI_STREAK_PERIOD = 2
CRSI_PERCENT_LOOKBACK = 100
WEEKLY_EMA_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
RSI_OVERBOUGHT = 90
RSI_OVERSOLD = 10
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_streak_rsi(close, period):
    """Calculate RSI based on consecutive up/down days streak"""
    # Calculate streak: consecutive up days = +1, down days = -1
    delta = np.diff(close, prepend=close[0])
    streak = np.zeros_like(close)
    current_streak = 0
    
    for i in range(len(close)):
        if delta[i] > 0:
            current_streak += 1
        elif delta[i] < 0:
            current_streak -= 1
        else:
            current_streak = 0
        streak[i] = current_streak
    
    # Calculate RSI on streak values
    abs_streak = np.abs(streak)
    up_streak = np.where(streak > 0, streak, 0)
    down_streak = np.where(streak < 0, -streak, 0)
    
    avg_up = pd.Series(up_streak).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_down = pd.Series(down_streak).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    rs = np.where(avg_down != 0, avg_up / avg_down, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_percent_rank(close, lookback):
    """Calculate percentile rank of current value over lookback period