#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining weekly pivot points and daily volume confirmation.
# Uses weekly pivot levels as dynamic support/resistance: long when price breaks above weekly R1 with high volume,
# short when breaks below weekly S1 with high volume.
# Weekly pivot provides institutional reference points, volume confirms breakout strength.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Weekly pivots adapt to changing volatility and provide meaningful levels in both bull and bear markets.

name = "exp_13835_6h_weekly_pivot_daily_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1  # Use previous week's data for pivot calculation
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_pivot_points(high, low, close):
    """Calculate classic pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H"""
    p = (high + low + close) / 3.0
    r1 = 2 * p - low
    s1 = 2 * p - high
    return p, r1, s1

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for pivot points ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    p_weekly, r1_weekly, s1_weekly = calculate_pivot_points(high_weekly, low_weekly, close_weekly)
    
    # Align weekly pivot points to 6h timeframe
    p_weekly_aligned = align_htf_to_ltf(prices, df_weekly, p_weekly)
    r1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r1_weekly)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s1_weekly)
    
    # Load daily data for volume confirmation ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily volume moving average
    volume_daily = df_daily['volume'].values
    volume_ma_daily = pd.Series(volume_daily).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # Align daily volume MA to 6h timeframe
    volume_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, volume_ma_daily)
    
    # 6h data for price and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(p_weekly_aligned[i]) or np.isnan(r1_weekly_aligned[i]) or np.isnan(s1_weekly_aligned[i]) or \
           np.isnan(volume_ma_daily_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation (using daily average)
        volume_ok = volume[i] > (volume_ma_daily_aligned[i] * VOLUME_THRESHOLD)
        
        # Breakout signals
        long_signal = volume_ok and close[i] > r1_weekly_aligned[i]
        short_signal = volume_ok and close[i] < s1_weekly_aligned[i]
        
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
            # Exit long on close below weekly pivot point
            if close[i] < p_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on close above weekly pivot point
            if close[i] > p_weekly_aligned[i]:
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

# Hypothesis: 6h strategy using Ichimoku Cloud from daily timeframe for trend filtering
# and 6h price action for entry timing. Long when price breaks above 6h resistance
# with price above daily Kumo (cloud) and TK cross bullish, short when breaks below
# 6h support with price below daily Kumo and TK cross bearish. Ichimoku provides
# multi-dimensional trend, support/resistance, and momentum in one indicator,
# effective in both trending and ranging markets. Designed for 50-150 total trades
# over 4 years (12-37/year) to minimize fee drag.

name = "exp_13835_6h_ichimoku_daily_tk_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TK_PERIOD = 9
KJ_PERIOD = 26
SENB_B_PERIOD = 52
DONCHIAN_ENTRY_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (HH + LL)/2 for TK_PERIOD
    tenkan_sen = (pd.Series(high).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).max() +
                  pd.Series(low).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).min()) / 2
    # Kijun-sen (Base Line): (HH + LL)/2 for KJ_PERIOD
    kijun_sen = (pd.Series(high).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).max() +
                 pd.Series(low).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted forward KJ_PERIOD
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (HH + LL)/2 for SENB_B_PERIOD shifted forward KJ_PERIOD
    senkou_span_b = ((pd.Series(high).rolling(window=SENB_B_PERIOD, min_periods=SENB_B_PERIOD).max() +
                      pd.Series(low).rolling(window=SENB_B_PERIOD, min_periods=SENB_B_PERIOD).min()) / 2)
    # Chikou Span (Lagging Span): close shifted back KJ_PERIOD
    chikou_span = pd.Series(close)
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values, chikou_span.values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Ichimoku ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily Ichimoku
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high_daily, low_daily, close_daily)
    
    # Align daily Ichimoku to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_daily, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_daily, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_daily, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_daily, senkou_b)
    chikou_aligned = align_htf_to_ltf(prices, df_daily, chikou)
    
    # 6h data for entry signals
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian channels for entry triggers
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_ENTRY_PERIOD, min_periods=DONCHIAN_ENTRY_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_ENTRY_PERIOD, min_periods=DONCHIAN_ENTRY_PERIOD).min().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(KJ_PERIOD, SENB_B_PERIOD, DONCHIAN_ENTRY_PERIOD, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or np.isnan(senkou_a_aligned[i]) or \
           np.isnan(senkou_b_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(volume_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine Kumo (cloud) top and bottom
        kumo_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        kumo_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # TK cross conditions
        tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
        tk_bearish = tenkan_aligned[i] < kijun_aligned[i]
        
        # Price relative to Kumo
        price_above_kumo = close[i] > kumo_top
        price_below_kumo = close[i] < kumo_bottom
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Entry signals
        long_signal = volume_ok and price_above_kumo and tk_bullish and close[i] > donchian_high[i]
        short_signal = volume_ok and price_below_kumo and tk_bearish and close[i] < donchian_low[i]
        
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
            # Exit long on TK cross bearish or price breaks below Kumo bottom
            if tk_bearish or close[i] < kumo_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on TK cross bullish or price breaks above Kumo top
            if tk_bullish or close[i] > kumo_top:
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

# Hypothesis: 6s strategy using Elder Ray Index (Bull Power/Bear Power) with 12h EMA trend filter
# and volume confirmation. Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 and rising, Bear Power < 0, price above 12h EMA20, and high volume.
# Short when Bear Power < 0 and falling, Bull Power > 0, price below 12h EMA20, and high volume.
# Elder Ray measures bull/bear power relative to EMA, effective in both trending and ranging markets.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drift.

name = "exp_13835_6h_elderray12h_ema20_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_RAY_EMA = 13
TREND_EMA = 20
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
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for EMA trend filter ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, TREND_EMA)
    
    # Align 12h EMA to 6h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 6h data for Elder Ray and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Elder Ray components (EMA13)
    ema_13 = calculate_ema(close, ELDER_RAY_EMA)
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ELDER_RAY_EMA, TREND_EMA, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_12h_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(volume_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Elder Ray conditions
        bull_power_positive = bull_power[i] > 0
        bear_power_negative = bear_power[i] < 0
        bull_power_rising = bull_power[i] > bull_power[i-1] if i > 0 else False
        bear_power_falling = bear_power[i] < bear_power[i-1] if i > 0 else False
        
        # Trend direction from 12h EMA
        above_ema = close[i] > ema_12h_aligned[i]
        below_ema = close[i] < ema_12h_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Entry signals
        long_signal = volume_ok and bull_power_positive and bear_power_negative and \
                      bull_power_rising and above_ema
        short_signal = volume_ok and bear_power_negative and bull_power_positive and \
                      bear_power_falling and below_ema
        
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
            # Exit long on Bear Power turning positive or price below 12h EMA
            if bear_power[i] >= 0 or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on Bull Power turning negative or price above 12h EMA
            if bull_power[i] <= 0 or close[i] > ema_12h_aligned[i]:
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

# Hypothesis: 6h strategy combining ADX trend strength with Williams %R momentum.
# Uses ADX(14) > 25 to identify trending markets, then Williams %R(14) for entry:
# Long when ADX > 25, %R crosses above -50 from below, and price above 200 EMA.
# Short when ADX > 25, %R crosses below -50 from above, and price below 200 EMA.
# ADX filters for strong trends, Williams %R provides timely entries within trends.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "exp_13835_6h_adx_williamsr_200ema_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
WILLIAMS_R_PERIOD = 14
EMA_FILTER = 200
ADX_THRESHOLD = 25
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_williams_r(high, low, close, period):
    """Calculate Williams %R"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero
    williams_r = williams_r.fillna(0)
    return williams_r.values

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smoothing
    tr_smoothed = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    dm_plus_smoothed = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    dm_minus_smoothed = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smoothed / tr_smoothed
    di_minus = 100 * dm_minus_smoothed / tr_smoothed
    
    # DX and ADX
    dx = np.abs((di_plus - di_minus) / (di_plus + di_minus)) * 100
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    return adx.values

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 6h data for indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R
    williams_r = calculate_williams_r(high, low, close, WILLIAMS_R_PERIOD)
    
    # Calculate ADX
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # Calculate EMA filter
    ema_filter = calculate_ema(close, EMA_FILTER)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD, WILLIAMS_R_PERIOD, EMA_FILTER, VOLUME_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(adx[i]) or np.isnan(williams_r[i]) or np.isnan(ema_filter[i]) or \
           np.isnan(volume_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # ADX trend strength
        strong_trend = adx[i] > ADX_THRESHOLD
        
        # Williams %R conditions
        wr_above_50 = williams_r[i] > -50
        wr_below_50 = williams_r[i] < -50
        wr_cross_up = (williams_r[i] > -50) and (williams_r[i-1] <= -50) if i > 0 else False
        wr_cross_down = (williams_r[i] < -50) and (williams_r[i-1] >= -50) if i > 0 else False
        
        # EMA filter
        price_above_ema = close[i] > ema_filter[i]
        price_below_ema = close[i] < ema_filter[i]
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Entry signals
        long_signal = strong_trend and wr_cross_up and price_above_ema and volume_ok
        short_signal = strong_trend and wr_cross_down and price_below_ema and volume_ok
        
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
            # Exit long on ADX weakening or Williams %R crossing below -50
            if adx[i] < ADX_THRESHOLD or wr_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on ADX weakening or Williams %R crossing above -50
            if adx[i] < ADX_THRESHOLD or wr_cross_up:
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

# Hypothesis: 6h strategy using Donchian breakout with weekly ADX trend filter and volume confirmation.
# Uses weekly ADX > 25 to identify strong trends, then 6h Donchian(20) breakouts for entry.
# Long when weekly ADX > 25, price breaks above 6h Donchian upper band with high volume.
# Short when weekly ADX > 25, price breaks below 6h Donchian lower band with high volume.
# Weekly ADX provides robust trend filter that adapts to changing market conditions,
# Donchian channels provide clear breakout levels, volume confirms breakout strength.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "exp_13835_6h_donchian20_weekly_adx_vol_v1"
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

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) -