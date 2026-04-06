#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h regime filter + volume confirmation.
# Elder Ray measures bull/bear power via EMA(13) divergence.
# 12h regime filter uses ADX(14) > 25 for trending markets (follow Elder Ray) and ADX < 20 for ranging (fade extremes).
# Volume > 1.5x average confirms institutional participation.
# Designed for low trade frequency to avoid fee drag in 6h timeframe.

name = "exp_13587_6h_elder_ray_12h_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_RAY_EMA = 13
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
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
    
    # Smooth TR, DM+, DM-
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for regime filter ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h ADX for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, ADX_PERIOD)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema_13 = calculate_ema(close, ELDER_RAY_EMA)
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ELDER_RAY_EMA, ADX_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_13[i]) or np.isnan(adx_12h_aligned[i]) or np.isnan(volume_ma[i]):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Regime filter from 12h ADX
        trending_market = adx_12h_aligned[i] > ADX_TREND_THRESHOLD
        ranging_market = adx_12h_aligned[i] < ADX_RANGE_THRESHOLD
        
        # Elder Ray signals with regime filter
        # In trending markets: follow Elder Ray direction
        # In ranging markets: fade Elder Ray extremes (mean reversion)
        long_signal = False
        short_signal = False
        
        if trending_market:
            # Follow the trend: long when bull power positive, short when bear power negative
            long_signal = volume_ok and (bull_power[i] > 0)
            short_signal = volume_ok and (bear_power[i] < 0)
        elif ranging_market:
            # Fade extremes: long when bear power extremely negative (oversold),
            # short when bull power extremely positive (overbought)
            long_signal = volume_ok and (bear_power[i] < -np.std(bear_power[max(0,i-50):i+1]) * 1.5)
            short_signal = volume_ok and (bull_power[i] > np.std(bull_power[max(0,i-50):i+1]) * 1.5)
        
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
            # Exit long when Elder Ray turns negative or stop loss hit
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when Elder Ray turns positive or stop loss hit
            if bear_power[i] >= 0:
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

# Hypothesis: 6h Camarilla pivot levels from 1d: fade at R3/S3, breakout continuation at R4/S4.
# Camarilla levels provide intraday support/resistance based on previous day's range.
# Fade at R3/S3 (80% probability of reversal) and break at R4/S4 (strong continuation).
# Volume > 1.3x average confirms institutional participation.
# Designed for low trade frequency to avoid fee drag in 6h timeframe.

name = "exp_13587_6h_camarilla1d_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.3
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    
    # Load 1d data for Camarilla pivots ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    range_1d = high_1d - low_1d
    camarilla_r4 = close_1d + (range_1d * 1.1 / 2)
    camarilla_r3 = close_1d + (range_1d * 1.1 / 4)
    camarilla_s3 = close_1d - (range_1d * 1.1 / 4)
    camarilla_s4 = close_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma[i]):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Camarilla signals with volume confirmation
        # Fade at R3/S3 (80% probability of reversal)
        fade_long = volume_ok and (close[i] <= camarilla_s3_aligned[i]) and (close[i] > camarilla_s4_aligned[i])
        fade_short = volume_ok and (close[i] >= camarilla_r3_aligned[i]) and (close[i] < camarilla_r4_aligned[i])
        
        # Breakout continuation at R4/S4 (strong momentum)
        breakout_long = volume_ok and (close[i] > camarilla_r4_aligned[i])
        breakout_short = volume_ok and (close[i] < camarilla_s4_aligned[i])
        
        # Generate signals
        if position == 0:
            if fade_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif fade_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on fade signal at R3 or stop loss
            if close[i] >= camarilla_r3_aligned[i] or close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on fade signal at S3 or stop loss
            if close[i] <= camarilla_s3_aligned[i] or close[i] >= stop_price:
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

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation.
# Weekly pivot provides directional bias from higher timeframe.
# Donchian breakout captures momentum in direction of weekly bias.
# Volume > 1.4x average confirms institutional participation.
# Designed for low trade frequency to avoid fee drag in 6h timeframe.

name = "exp_13587_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.4
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    
    # Load weekly data for pivot direction ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    # Bias: above pivot = bullish, below pivot = bearish
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    bullish_bias = close_1w > weekly_pivot  # Bullish if close above pivot
    bearish_bias = close_1w < weekly_pivot  # Bearish if close below pivot
    
    # Align weekly bias to 6h timeframe
    bullish_bias_aligned = align_htf_to_ltf(prices, df_1w, bullish_bias)
    bearish_bias_aligned = align_htf_to_ltf(prices, df_1w, bearish_bias)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(bullish_bias_aligned[i]) or np.isnan(volume_ma[i]):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Donchian breakout signals with weekly pivot bias
        # Only take longs in bullish weekly bias, shorts in bearish weekly bias
        long_signal = volume_ok and (close[i] > donchian_high[i-1]) and bullish_bias_aligned[i]
        short_signal = volume_ok and (close[i] < donchian_low[i-1]) and bearish_bias_aligned[i]
        
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
            # Exit long on break below Donchian low or weekly bias change
            if close[i] < donchian_low[i] or not bullish_bias_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on break above Donchian high or weekly bias change
            if close[i] > donchian_high[i] or not bearish_bias_aligned[i]:
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

# Hypothesis: 6h Ichimoku: TK cross + cloud filter from 1d (TIER 8 in program.md).
# Ichimoku provides comprehensive trend, momentum, and support/resistance.
# TK cross (Tenkan/Kijun) signals momentum shift.
# Cloud (Senkou Span A/B) from 1d acts as major support/resistance filter.
# Only take trades aligned with daily cloud (price above cloud for longs, below for shorts).
# Volume > 1.3x average confirms institutional participation.
# Designed for low trade frequency to avoid fee drag in 6h timeframe.

name = "exp_13587_6h_ichimoku1d_tk_cross_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TK_PERIOD = 9  # Tenkan-sen period
KJ_PERIOD = 26 # Kijun-sen period
SS_PERIOD = 52 # Senkou Span period
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.3
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    
    # Load 1d data for Ichimoku cloud ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    tenkan_sen = (pd.Series(high_1d).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).max() + 
                  pd.Series(low_1d).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    kijun_sen = (pd.Series(high_1d).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).max() + 
                 pd.Series(low_1d).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(1)  # Shifted for cloud
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    senkou_span_b = ((pd.Series(high_1d).rolling(window=SS_PERIOD, min_periods=SS_PERIOD).max() + 
                      pd.Series(low_1d).rolling(window=SS_PERIOD, min_periods=SS_PERIOD).min()) / 2).shift(1)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Tenkan-sen and Kijun-sen for TK cross
    tenkan_sen_6h = (pd.Series(high).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).max() + 
                     pd.Series(low).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).min()) / 2
    kijun_sen_6h = (pd.Series(high).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).max() + 
                    pd.Series(low).rolling(window=KJ_PERIOD, min_periods=KJ_PERIOD).min()) / 2
    
    # TK cross signals: Tenkan crosses above/below Kijun
    tk_cross_up = (tenkan_sen_6h > kijun_sen_6h) & (tenkan_sen_6h.shift(1) <= kijun_sen_6h.shift(1))
    tk_cross_down = (tenkan_sen_6h < kijun_sen_6h) & (tenkan_sen_6h.shift(1) >= kijun_sen_6h.shift(1))
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TK_PERIOD, KJ_PERIOD, SS_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(volume_ma[i])):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Cloud filter: price above cloud = bullish, below cloud = bearish
        # Cloud top = max(Senkou Span A, Senkou Span B)
        # Cloud bottom = min(Senkou Span A, Senkou Span B)
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Ichimoku signals with cloud filter and volume confirmation
        long_signal = volume_ok and tk_cross_up[i] and price_above_cloud
        short_signal = volume_ok and tk_cross_down[i] and price_below_cloud
        
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
            # Exit long on TK cross down or price drops below cloud
            if tk_cross_down[i] or close[i] < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on TK cross up or price rises above cloud
            if tk_cross_up[i] or close[i] > cloud_top:
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

# Hypothesis: 6h ADX + Williams Alligator combination.
# ADX(14) > 25 indicates strong trend (trend following mode).
# Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and entry points.
# Only trade when Alligator is "awake" (lines not intertwined) and ADX confirms trend strength.
# Volume > 1.3x average confirms institutional participation.
# Designed for low trade frequency to avoid fee drag in 6h timeframe.

name = "exp_13587_6h_adx_alligator_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ADX_PERIOD = 14
ADX_THRESHOLD = 25
ALLIGATOR_JAW = 13  # Smoothed Median (13-period SMMA, 8-period shift)
ALLIGATOR_TEETH = 8  # Smoothed Median (8-period SMMA, 5-period shift)
ALLIGATOR_LIPS = 5   # Smoothed Median (5-period SMMA, 3-period shift)
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.3
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_smma(data, period):
    """Calculate Smoothed Moving Average (SMMA)"""
    sma = pd.Series(data).rolling(window=period, min_periods=period).mean()
    # SMMA formula: SMMA_t = (SMMA_{t-1} * (period-1) + close_t) / period
    smma = np.full_like(data, np.nan, dtype=float)
    if len(data) >=