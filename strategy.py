#!/usr/bin/env python3
"""
Experiment #9647: 6h Elder Ray Power + Volume Spike + Regime Filter.
Hypothesis: Elder Ray (Bull Power and Bear Power) identifies institutional buying/selling pressure.
Combined with volume spikes for confirmation and regime filtering (ADX for trend strength),
this strategy captures momentum in both bull and bear markets. In low volatility (ADX<25),
it fades extreme power readings; in high volatility (ADX>=25), it follows the power direction.
Targets 75-150 total trades over 4 years (19-38/year) to balance opportunity and cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9647_6h_elder_ray_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
EMA_PERIOD = 13
VOLUME_SPIKE_MULTIPLIER = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 25
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_true_range(high, low, close):
    """Calculate True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First value
    return tr

def calculate_adx(high, low, close, period):
    """Calculate ADX using Wilder's smoothing"""
    tr = calculate_true_range(high, low, close)
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr = calculate_true_range(high, low, close)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA for Elder Ray calculation
    ema = calculate_ema(close, EMA_PERIOD)
    
    # Bull Power = High - EMA, Bear Power = Low - EMA
    bull_power = high - ema
    bear_power = low - ema
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX for regime filtering
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_PERIOD, ADX_PERIOD, ATR_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if any indicator not available
        if np.isnan(ema[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_ma[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Regime filter: ADX < 25 for mean reversion, ADX > 25 for trend following
        low_volatility = adx[i] < ADX_THRESHOLD   # Ranging market
        high_volatility = adx[i] >= ADX_THRESHOLD  # Trending market
        
        # Mean reversion signals (ADX < 25): fade extreme power readings
        mean_rev_long = low_volatility and volume_spike and bear_power[i] < -np.std(bear_power[max(0, i-50):i+1]) * 2.0
        mean_rev_short = low_volatility and volume_spike and bull_power[i] > np.std(bull_power[max(0, i-50):i+1]) * 2.0
        
        # Trend following signals (ADX >= 25): follow power direction
        trend_long = high_volatility and volume_spike and bull_power[i] > 0 and bear_power[i] < 0
        trend_short = high_volatility and volume_spike and bull_power[i] < 0 and bear_power[i] > 0
        
        # Entry conditions
        long_entry = mean_rev_long or trend_long
        short_entry = mean_rev_short or trend_short
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
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
"""
Experiment #9647: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation.
Hypothesis: Donchian(20) breakouts on 6h timeframe, filtered by weekly pivot direction (from 1w data),
provide high-probability trend continuation signals. Volume confirmation ensures institutional participation.
Works in bull markets (breakouts above weekly pivot) and bear markets (breakdowns below weekly pivot).
Targets 100-200 total trades over 4 years (25-50/year) with selective entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9647_6h_donchian20_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels: upper = max(high, period), lower = min(low, period)"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for weekly pivot direction)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w pivot points (using previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point and support/resistance levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    # Weekly support/resistance levels (using standard pivot formula)
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align 1w levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    upper, lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Weekly pivot direction filter
        # Bullish bias: price above weekly pivot
        bullish_bias = close[i] > pivot_1w_aligned[i]
        # Bearish bias: price below weekly pivot
        bearish_bias = close[i] < pivot_1w_aligned[i]
        
        # Breakout signals with weekly pivot confirmation
        # Long: Donchian breakout above upper channel + bullish bias + volume spike
        long_breakout = (high[i] > upper[i]) and bullish_bias and volume_spike
        # Short: Donchian breakdown below lower channel + bearish bias + volume spike
        short_breakout = (low[i] < lower[i]) and bearish_bias and volume_spike
        
        # Generate signals
        if position == 0:
            if long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_breakout:
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
"""
Experiment #9647: 6h Elder Ray Power + Volume Spike + Regime Filter.
Hypothesis: Elder Ray (Bull Power and Bear Power) identifies institutional buying/selling pressure.
Combined with volume spikes for confirmation and regime filtering (ADX for trend strength),
this strategy captures momentum in both bull and bear markets. In low volatility (ADX<25),
it fades extreme power readings; in high volatility (ADX>=25), it follows the power direction.
Targets 75-150 total trades over 4 years (19-38/year) to balance opportunity and cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9647_6h_elder_ray_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
EMA_PERIOD = 13
VOLUME_SPIKE_MULTIPLIER = 2.0
ADX_PERIOD = 14
ADX_THRESHOLD = 25
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_true_range(high, low, close):
    """Calculate True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First value
    return tr

def calculate_adx(high, low, close, period):
    """Calculate ADX using Wilder's smoothing"""
    tr = calculate_true_range(high, low, close)
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
    
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr = calculate_true_range(high, low, close)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA for Elder Ray calculation
    ema = calculate_ema(close, EMA_PERIOD)
    
    # Bull Power = High - EMA, Bear Power = Low - EMA
    bull_power = high - ema
    bear_power = low - ema
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX for regime filtering
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_PERIOD, ADX_PERIOD, ATR_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if any indicator not available
        if np.isnan(ema[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_ma[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Regime filter: ADX < 25 for mean reversion, ADX > 25 for trend following
        low_volatility = adx[i] < ADX_THRESHOLD   # Ranging market
        high_volatility = adx[i] >= ADX_THRESHOLD  # Trending market
        
        # Calculate dynamic thresholds for extreme power readings
        # Use rolling standard deviation of power values
        lookback = min(50, i+1)
        if lookback >= 10:
            bull_std = np.std(bull_power[max(0, i-lookback):i+1])
            bear_std = np.std(bear_power[max(0, i-lookback):i+1])
        else:
            bull_std = 1.0
            bear_std = 1.0
        
        # Mean reversion signals (ADX < 25): fade extreme power readings
        mean_rev_long = low_volatility and volume_spike and bear_power[i] < -bear_std * 2.0
        mean_rev_short = low_volatility and volume_spike and bull_power[i] > bull_std * 2.0
        
        # Trend following signals (ADX >= 25): follow power direction
        trend_long = high_volatility and volume_spike and bull_power[i] > 0 and bear_power[i] < 0
        trend_short = high_volatility and volume_spike and bull_power[i] < 0 and bear_power[i] > 0
        
        # Entry conditions
        long_entry = mean_rev_long or trend_long
        short_entry = mean_rev_short or trend_short
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
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
"""
Experiment #9647: 6h Ichimoku Cloud + TK Cross + Volume Filter.
Hypothesis: Ichimoku system provides comprehensive trend, support/resistance, and momentum signals.
TK line (Tenkan/Kijun) cross acts as entry trigger, filtered by cloud (Senkou Span) position and volume.
Works in bull markets (price above cloud, bullish TK cross) and bear markets (price below cloud, bearish TK cross).
Targets 80-160 total trades over 4 years (20-40/year) with multi-factor confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9647_6h_ichimoku_tk_cross_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_PERIOD = 52
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    tenkan = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
              pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    kijun = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
             pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods
    senkou_b = (pd.Series(high).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).max() + 
                pd.Series(low).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).min()) / 2
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    chikou = pd.Series(close)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values, chikou.values

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
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku Cloud components
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (need Senkou B data)
    start = max(SENKOU_PERIOD, KIJUN_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if any indicator not available
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Cloud position (determine trend)
        # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
        cloud_top = np.maximum(senkou_a[i], senkou_b[i])
        cloud_bottom = np.minimum(senkou_a[i], senkou_b[i])
        
        # Price above cloud = bullish bias
        price_above_cloud = close[i] > cloud_top
        # Price below cloud = bearish bias
        price_below_cloud = close[i] < cloud_bottom
        # Price inside cloud = neutral (no strong trend)
        price_in_cloud = (close[i] >= cloud_bottom) and (close[i] <= cloud_top)
        
        # TK Cross signals
        # Bullish TK cross: Tenkan crosses above Kijun
        tk_bullish_cross = (tenkan[i] > kijun[i]) and (tenkan[i-1] <= kijun[i-1])
        # Bearish TK cross: Tenkan crosses below Kijun
        tk_bearish_cross = (tenkan[i] < kijun[i]) and (tenkan[i-1] >= kijun[i-1])
        
        # Entry conditions
        # Long: Bullish TK cross + price above cloud + volume spike
        long_entry = tk_bullish_cross and price_above_cloud and volume_spike
        # Short: Bearish TK cross + price below cloud + volume spike
        short_entry = tk_bearish_cross and price_below_cloud and volume_spike
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
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
"""
Experiment #9647: 6h ADX + Williams Alligator Combination.
Hypothesis: Williams Alligator (Jaw, Teeth, Lips) identifies trend absence/presence and direction.
Combined with ADX for trend strength filtering, this strategy avoids whipsaws in ranging markets
and captures strong trends. In non-trending markets (ADX<20), it stays flat. In trending markets
(ADX>=20), it follows Alligator alignment: long when Lips>Teeth>Jaw, short when Lips<Teeth<Jaw.
Targets 60-120 total trades over 4 years (15-30/year) to minimize false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9647_6h_adx_alligator_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13   # Smoothed Median (13-period SMMA, 8-period shift)
ALLIGATOR_TEETH_PERIOD = 8  # Smoothed Median (8-period SMMA, 5-period shift)
ALLIGATOR_LIPS_PERIOD = 5   # Smoothed Median (5-period SMMA, 3-period shift)
ADX_PERIOD = 14
ADX_THRESHOLD = 20          # Only trade when ADX >= 20 (trending market)
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def smoothed_moving_average(series, period):
    """Calculate Smoothed Moving Average (SMMA) - similar to Wilder's smoothing"""
    # SMMA is essentially EMA with alpha = 1/period
    return pd.Series(series).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def calculate_alligator(high, low, close):
    """Calculate Williams Alligator lines"""
    # Typical Price = (High + Low + Close) / 3
    typical_price = (high + low + close) / 3
    
    # Jaw: 13-period SMMA of typical price, shifted 8 bars forward
    jaw = smoothed_moving_average(typical_price, ALLIGATOR_JAW_PERIOD)
    # Teeth: 8-period SMMA of typical price, shifted 5 bars forward
    teeth = smoothed_moving_average(typical_price, ALLIGATOR_TEETH_PERIOD)
    # Lips: 5-period SMMA of typical price, shifted 3 bars forward
    lips = smoothed_moving_average(typical_price, ALLIGATOR_LIPS_PERIOD)
    
    return jaw, teeth, lips

def calculate_adx(high, low, close, period):
    """Calculate ADX using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    plus_dm = np.where((high - np.roll(high, 1