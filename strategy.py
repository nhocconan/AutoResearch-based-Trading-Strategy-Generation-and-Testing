#!/usr/bin/env python3
"""
Experiment #7591: 6h Donchian(20) breakout with 1-day trend filter and volume confirmation.
Hypothesis: In bull markets (price > 1d EMA50), go long on breakout above 6h Donchian upper.
In bear markets (price < 1d EMA50), go short on breakdown below 6h Donchian lower.
Volume must be above 1.5x average to confirm breakout strength.
Targets 100-200 trades over 4 years (25-50/year) with strict breakout conditions.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7591_6h_donchian20_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_TREND = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d_50 = pd.Series(close_1d).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_TREND, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1d_50_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market regime
        bull_regime = close[i] > ema_1d_50_aligned[i]   # price above 1d EMA50
        bear_regime = close[i] < ema_1d_50_aligned[i]   # price below 1d EMA50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions
        upper_breakout = (high[i] > highest_high[i-1]) and (i-1 >= 0) and not np.isnan(highest_high[i-1])
        lower_breakout = (low[i] < lowest_low[i-1]) and (i-1 >= 0) and not np.isnan(lowest_low[i-1])
        
        # Entry conditions
        long_entry = bull_regime and upper_breakout and volume_confirmed
        short_entry = bear_regime and lower_breakout and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
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
Experiment #7591: 6h Camarilla Pivot Reversal Strategy.
Hypothesis: Camarilla pivot levels from daily timeframe act as strong support/resistance.
Price tends to reverse at R3/S3 levels and break through R4/S4 levels with momentum.
In ranging markets (ADX < 25), fade extremes at R3/S3. In trending markets (ADX >= 25),
breakouts at R4/S4 continue the trend. Volume confirmation required for entries.
Targets 100-200 trades over 4 years (25-50/year).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7591_6h_camarilla_pivot_adx_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_LOOKBACK = 1  # Use previous day's OHLC
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    # Formula based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and ranges
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    r4 = close_1d + range_hl * 1.1 / 2
    r3 = close_1d + range_hl * 1.1 / 4
    s3 = close_1d - range_hl * 1.1 / 4
    s4 = close_1d - range_hl * 1.1 / 2
    
    # Align to 6h timeframe (shifted by 1 day for no look-ahead)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ADX for trend detection
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    # Pad arrays to match length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Smoothed values
    atr_period = ADX_PERIOD
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=atr_period, adjust=False).mean() / 
                     pd.Series(tr).ewm(span=atr_period, adjust=False).mean())
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=atr_period, adjust=False).mean() / 
                      pd.Series(tr).ewm(span=atr_period, adjust=False).mean())
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=atr_period, adjust=False).mean().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr_atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr_atr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ADX_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(adx[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1 and close[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine market regime
        is_trending = adx[i] >= ADX_TREND_THRESHOLD
        is_ranging = adx[i] < ADX_TREND_THRESHOLD
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trading logic based on market regime
        if is_ranging and volume_confirmed:
            # In ranging markets, fade extremes at R3/S3
            if close[i] <= s3_aligned[i]:
                # Potential long at S3 support
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif close[i] >= r3_aligned[i]:
                # Potential short at R3 resistance
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif is_trending and volume_confirmed:
            # In trending markets, breakout continuation at R4/S4
            if close[i] > r4_aligned[i]:
                # Breakout above R4 - go long
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif close[i] < s4_aligned[i]:
                # Breakdown below S4 - go short
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        else:
            # No clear signal
            signals[i] = 0.0
        
        # Maintain position
        if position == 1 and signals[i] == 0:
            signals[i] = SIGNAL_SIZE
        elif position == -1 and signals[i] == 0:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #7591: 6h Elder Ray Index with ADX Regime Filter.
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) 
measures bull/bear strength relative to trend. In strong trends (ADX > 25),
trade in direction of Elder Ray with EMA crossover confirmation. 
In ranging markets (ADX < 20), fade extreme Elder Ray readings.
Volume confirmation required for all entries. Targets 100-180 trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7591_6h_elder_ray_adx_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
EMA_ELDER = 13
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
ADX_RANGE_THRESHOLD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - daily trend filter for context
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for higher timeframe trend bias
    close_1d = df_1d['close'].values
    ema_1d_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=EMA_ELDER, adjust=False, min_periods=EMA_ELDER).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # ADX for trend detection
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    # Pad arrays
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Smoothed values for DI calculation
    atr_period = ADX_PERIOD
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=atr_period, adjust=False).mean() / 
                     pd.Series(tr).ewm(span=atr_period, adjust=False).mean())
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=atr_period, adjust=False).mean() / 
                      pd.Series(tr).ewm(span=atr_period, adjust=False).mean())
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=atr_period, adjust=False).mean().values
    
    # EMA crossover for entry confirmation (fast EMA8, slow EMA21)
    ema_8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr_atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr_atr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_ELDER, ADX_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, 21) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1d_50_aligned[i]) or np.isnan(adx[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1 and close[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine market regime from ADX
        is_strong_trend = adx[i] >= ADX_TREND_THRESHOLD
        is_ranging = adx[i] < ADX_RANGE_THRESHOLD
        # Transition zone (20 <= ADX < 25) - no clear regime, stay flat
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # EMA crossover signals
        ema_bullish = ema_8[i] > ema_21[i]
        ema_bearish = ema_8[i] < ema_21[i]
        
        # Trading logic based on regime
        if is_strong_trend and volume_confirmed:
            # In strong trends, trade with Elder Ray and EMA confirmation
            if bull_power[i] > 0 and bear_power[i] < 0 and ema_bullish and close[i] > ema_1d_50_aligned[i]:
                # Strong bullish alignment: positive bull power, negative bear power, EMA bullish, above daily trend
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif bull_power[i] < 0 and bear_power[i] > 0 and ema_bearish and close[i] < ema_1d_50_aligned[i]:
                # Strong bearish alignment: negative bull power, positive bear power, EMA bearish, below daily trend
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif is_ranging and volume_confirmed:
            # In ranging markets, fade extreme Elder Ray readings
            if bull_power[i] > np.percentile(bull_power[max(0, i-100):i+1], 90) and ema_bearish:
                # Extremely bullish reading in range - fade with short
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif bear_power[i] < np.percentile(bear_power[max(0, i-100):i+1], 10) and ema_bullish:
                # Extremely bearish reading in range - fade with long
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        else:
            # Transition zone or no confirmation - stay flat
            signals[i] = 0.0
        
        # Maintain position
        if position == 1 and signals[i] == 0:
            signals[i] = SIGNAL_SIZE
        elif position == -1 and signals[i] == 0:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #7591: 6h Donchian Breakout with Volume-Weighted RPI and ADX Filter.
Hypothesis: Donchian channel breakouts combined with Volume-Weighted Relative Price Index 
(RPI = (Close - Min)/(Max - Min) * Volume) provide high-probability entries. 
ADX > 25 filters for trending markets only. Volume must exceed 2x average for breakout 
validation. Targets 80-150 trades over 4 years (20-38/year).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7591_6h_donchian_rpi_adx_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
RPI_PERIOD = 14
ADX_PERIOD = 14
ADX_THRESHOLD = 25
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0  # Higher threshold for quality breakouts
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - daily trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA100 for higher timeframe trend bias
    close_1d = df_1d['close'].values
    ema_1d_100 = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_1d_100_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_100)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume-Weighted Relative Price Index (VW-RPI)
    # RPI = (Close - Lowest Low) / (Highest High - Lowest Low) * Volume
    price_range = highest_high - lowest_low
    # Avoid division by zero
    price_range_safe = np.where(price_range == 0, 1, price_range)
    rpi_raw = (close - lowest_low) / price_range_safe * volume
    # Normalize RPI for better interpretation
    rpi = pd.Series(rpi_raw).ewm(span=RPI_PERIOD, adjust=False, min_periods=RPI_PERIOD).mean().values
    
    # ADX for trend detection
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    # Pad arrays to match length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Smoothed values for DI calculation
    atr_period = ADX_PERIOD
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=atr_period, adjust=False).mean() / 
                     pd.Series(tr).ewm(span=atr_period, adjust=False).mean())
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=atr_period, adjust=False).mean() / 
                      pd.Series(tr).ewm(span=atr_period, adjust=False).mean())
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=atr_period, adjust=False).mean().values
    
    # Volume moving average for breakout validation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr_atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr_atr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, RPI_PERIOD, ADX_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1d_100_aligned[i]) or np.isnan(adx[i]) or np.isnan(rpi[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1 and close[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine market regime
        is_trending = adx[i] >= ADX_THRESHOLD
        
        # Volume confirmation for breakout
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions with VW-RPI confirmation
        # Long: Break above Donchian upper with high VW-RPI (strong buying pressure)
        upper_breakout = (high[i] > highest_high[i-1]) and (i-1 >= 0) and not np.isnan(highest_high[i-1])
        # Short: Break below Donchian lower with low VW-RPI (strong selling pressure)
        lower_breakout = (low[i] < lowest_low[i-1]) and (i-1 >= 0) and not np.isnan(lowest_low[i-1])
        
        # VW-RPI thresholds (extreme values indicate strong pressure)
        rpi_high = np.percentile(rpi[max(0, i-100):i+1], 80) if i >= 100 else np.percentile(rpi[max(0, i-20):i+1], 80)
        rpi_low = np.percentile(rpi[max(0, i-100):i+1], 20) if i >= 100 else np.percentile(rpi[max(0, i-20):i+1], 20)
        
        # Entry conditions
        long_entry = (is_trending and 
                     upper_breakout and 
                     volume_confirmed and 
                     rpi[i] > rpi_high and  # Strong buying pressure
                     close[i] > ema_1d_100_aligned[i])  # Above daily trend
        
        short_entry = (is_trending and 
                      lower_breakout and 
                      volume_confirmed and 
                      rpi[i] < rpi_low and  # Strong selling pressure
                      close[i] < ema_1d_100_aligned[i])  # Below daily trend
        
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

--- End of file ---