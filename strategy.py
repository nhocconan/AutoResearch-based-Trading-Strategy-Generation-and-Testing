#!/usr/bin/env python3
"""
Experiment #7839: 6-hour Donchian breakout with 12-hour volume confirmation and 1-day trend filter.
Hypothesis: Price breaking beyond 20-period high/low on 6h with volume >1.5x 20-period MA and aligned 1d trend (EMA) captures sustained moves while avoiding whipsaw. The 1d trend filter provides directional bias from higher timeframe to reduce false breakouts in both bull and bear markets. Targets 100-200 trades over 4 years with controlled risk via ATR-based stops.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7839_6h_donchian20_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
EMA_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Trend bias: above EMA = bullish, below EMA = bearish
    trend_bias_1d = np.where(close_1d > ema_1d, 1, -1)  # 1=bullish, -1=bearish
    trend_bias_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_bias_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Price channel (Donchian)
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
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, EMA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(trend_bias_1d_aligned[i]):
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
        
        # Determine market bias from 1d EMA
        bull_bias = trend_bias_1d_aligned[i] == 1   # 1d close above EMA
        bear_bias = trend_bias_1d_aligned[i] == -1  # 1d close below EMA
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions - require close beyond channel bands to avoid wicks
        upper_breakout = (close[i] > highest_high[i-1]) and (i-1 >= 0) and not np.isnan(highest_high[i-1])
        lower_breakout = (close[i] < lowest_low[i-1]) and (i-1 >= 0) and not np.isnan(lowest_low[i-1])
        
        # Entry conditions
        long_entry = bull_bias and upper_breakout and volume_confirmed
        short_entry = bear_bias and lower_breakout and volume_confirmed
        
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
Experiment #7839: 6-hour Camarilla pivot reversal with 12-hour volume confirmation.
Hypothesis: Price rejecting at Camarilla R3/S3 levels with volume >1.8x 20-period MA and aligned 12h trend (EMA) captures mean reversion moves in ranging markets, while breakouts at R4/S4 with volume and trend capture trend continuation. Works in both bull/bear regimes by adapting to price action at key levels. Targets 80-180 trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7839_6h_camarilla_reversal_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
EMA_PERIOD = 30
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Trend bias: above EMA = bullish, below EMA = bearish
    trend_bias_12h = np.where(close_12h > ema_12h, 1, -1)  # 1=bullish, -1=bearish
    trend_bias_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_bias_12h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous bar
    # Typical Price = (H + L + C) / 3
    typical_price = (high + low + close) / 3.0
    # Range = H - L
    range_hl = high - low
    
    # Camarilla levels
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    r4 = close + (range_hl * CAMARILLA_MULT / 2.0)
    r3 = close + (range_hl * CAMARILLA_MULT / 4.0)
    s3 = close - (range_hl * CAMARILLA_MULT / 4.0)
    s4 = close - (range_hl * CAMARILLA_MULT / 2.0)
    
    # Shift levels to get previous bar's levels (no look-ahead)
    r4_prev = np.roll(r4, 1)
    r3_prev = np.roll(r3, 1)
    s3_prev = np.roll(s3, 1)
    s4_prev = np.roll(s4, 1)
    r4_prev[0] = np.nan
    r3_prev[0] = np.nan
    s3_prev[0] = np.nan
    s4_prev[0] = np.nan
    
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
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(trend_bias_12h_aligned[i]):
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
        
        # Determine market bias from 12h EMA
        bull_bias = trend_bias_12h_aligned[i] == 1   # 12h close above EMA
        bear_bias = trend_bias_12h_aligned[i] == -1  # 12h close below EMA
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Fade at R3/S3 (mean reversion) - only in ranging conditions
        # Fade R3: price rejected at R3, go short
        fade_r3 = (close[i] < r3_prev[i]) and (close[i-1] >= r3_prev[i-1]) and not np.isnan(r3_prev[i]) and volume_confirmed
        # Fade S3: price rejected at S3, go long
        fade_s3 = (close[i] > s3_prev[i]) and (close[i-1] <= s3_prev[i-1]) and not np.isnan(s3_prev[i]) and volume_confirmed
        
        # Breakout at R4/S4 (trend continuation) - only with trend alignment
        # Breakout R4: price broke above R4, go long
        breakout_r4 = (close[i] > r4_prev[i]) and (close[i-1] <= r4_prev[i-1]) and not np.isnan(r4_prev[i]) and bull_bias and volume_confirmed
        # Breakout S4: price broke below S4, go short
        breakout_s4 = (close[i] < s4_prev[i]) and (close[i-1] >= s4_prev[i-1]) and not np.isnan(s4_prev[i]) and bear_bias and volume_confirmed
        
        # Entry conditions
        long_entry = fade_s3 or breakout_r4
        short_entry = fade_r3 or breakout_s4
        
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
Experiment #7839: 6-hour Elder Ray Index with 12-hour trend filter and volume confirmation.
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) identifies bull/bear strength. 
Enter long when Bull Power > 0 and rising with volume >1.5x MA and 12h uptrend (EMA50). 
Enter short when Bear Power > 0 and rising with volume >1.5x MA and 12h downtrend.
Uses 13-period EMA for Elder Ray as per original Alexander Elder definition. 
Targets 90-190 trades over 4 years with ATR-based stops.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7839_6h_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_EMA_PERIOD = 13
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
TREND_EMA_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=TREND_EMA_PERIOD, adjust=False, min_periods=TREND_EMA_PERIOD).mean().values
    
    # Trend bias: above EMA = bullish, below EMA = bearish
    trend_bias_12h = np.where(close_12h > ema_12h, 1, -1)  # 1=bullish, -1=bearish
    trend_bias_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_bias_12h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Elder Ray components
    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=ELDER_EMA_PERIOD, adjust=False, min_periods=ELDER_EMA_PERIOD).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = EMA13 - Low
    bear_power = ema13 - low
    
    # Rising condition: current value > previous value
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bear_power_rising = bear_power > np.roll(bear_power, 1)
    # Handle first element
    bull_power_rising[0] = False
    bear_power_rising[0] = False
    
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
    
    # Start from warmup period
    start = max(ELDER_EMA_PERIOD, VOLUME_MA_PERIOD, TREND_EMA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(trend_bias_12h_aligned[i]):
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
        
        # Determine market bias from 12h EMA
        bull_bias = trend_bias_12h_aligned[i] == 1   # 12h close above EMA
        bear_bias = trend_bias_12h_aligned[i] == -1  # 12h close below EMA
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        # Long: Bull Power > 0 AND rising AND volume confirmed AND 12h uptrend
        long_entry = (bull_power[i] > 0) and bull_power_rising[i] and volume_confirmed and bull_bias
        # Short: Bear Power > 0 AND rising AND volume confirmed AND 12h downtrend
        short_entry = (bear_power[i] > 0) and bear_power_rising[i] and volume_confirmed and bear_bias
        
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
Experiment #7839: 6-hour Williams Alligator with 12-hour trend filter and volume confirmation.
Hypothesis: Williams Alligator (Jaw=TEETH=LIPS SMMA) identifies trendless markets when lines are intertwined.
Enter long when Lips > Teeth > Jaw (bullish alignment) with price above all lines and volume >1.8x MA.
Enter short when Lips < Teeth < Jaw (bearish alignment) with price below all lines and volume >1.8x MA.
Uses smoothed moving averages (SMMA) as per Bill Williams definition. 
Targets 70-170 trades over 4 years with ATR-based stops.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7839_6h_williams_alligator_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13   # Smoothed MA (Blue line)
ALLIGATOR_TEETH_PERIOD = 8  # Smoothed MA (Red line)
ALLIGATOR_LIPS_PERIOD = 5   # Smoothed MA (Green line)
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
TREND_EMA_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def smma(series, period):
    """Smoothed Moving Average as used in Williams Alligator"""
    if len(series) == 0:
        return np.array([])
    result = np.full_like(series, np.nan, dtype=float)
    # First value is simple average
    if len(series) >= period:
        result[period-1] = np.mean(series[:period])
    # Subsequent values: (prev_smma * (period-1) + current_value) / period
    for i in range(period, len(series)):
        if not np.isnan(result[i-1]):
            result[i] = (result[i-1] * (period-1) + series[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=TREND_EMA_PERIOD, adjust=False, min_periods=TREND_EMA_PERIOD).mean().values
    
    # Trend bias: above EMA = bullish, below EMA = bearish
    trend_bias_12h = np.where(close_12h > ema_12h, 1, -1)  # 1=bullish, -1=bearish
    trend_bias_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_bias_12h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator components (using SMMA)
    jaw = smma(close, ALLIGATOR_JAW_PERIOD)    # Blue line
    teeth = smma(close, ALLIGATOR_TEETH_PERIOD) # Red line
    lips = smma(close, ALLIGATOR_LIPS_PERIOD)   # Green line
    
    # Alligator alignment conditions
    # Bullish: Lips > Teeth > Jaw
    bullish_aligned = (lips > teeth) & (teeth > jaw)
    # Bearish: Lips < Teeth < Jaw
    bearish_aligned = (lips < teeth) & (teeth < jaw)
    
    # Price position relative to Alligator lines
    price_above_all = (close > lips) & (close > teeth) & (close > jaw)
    price_below_all = (close < lips) & (close < teeth) & (close < jaw)
    
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
    
    # Start from warmup period
    start = max(ALLIGATOR_JAW_PERIOD, VOLUME_MA_PERIOD, TREND_EMA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(trend_bias_12h_aligned[i]):
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
        
        # Determine market bias from 12h EMA
        bull_bias = trend_bias_12h_aligned[i] == 1   # 12h close above EMA
        bear_bias = trend_bias_12h_aligned[i] == -1  # 12h close below EMA
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        # Long: Bullish Alligator alignment AND price above all lines AND volume confirmed AND 12h uptrend
        long_entry = bullish_aligned[i] and price_above_all[i] and volume_confirmed and bull_bias
        # Short: Bearish Alligator alignment AND price below all lines AND volume confirmed AND 12h downtrend
        short_entry = bearish_aligned[i] and price_below_all[i] and volume_confirmed and bear_bias
        
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

--- 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 === 0 ===