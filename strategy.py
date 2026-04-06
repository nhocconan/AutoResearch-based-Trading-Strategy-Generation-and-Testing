#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h trend filter and volume confirmation.
# Donchian channels identify breakout points where price moves beyond recent highs/lows,
# often signaling the start of a new trend. In volatile markets like crypto, breakouts
# from the 20-period Donchian channel capture significant moves. The 12h EMA(50) slope
# filters for trades in the direction of the higher timeframe trend, reducing false
# breakouts. Volume confirmation ensures breakouts are supported by participation.
# Works in bull markets (buy breakouts above upper channel) and bear markets (sell
# breakdowns below lower channel). Target: 25-50 trades/year by requiring breakouts
# beyond the 20-period channel, trend alignment, and volume >1.5x 20-period average.

name = "exp_13653_4h_donchian20_12h_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels: upper = max(high, period), lower = min(low, period)"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, TREND_EMA_PERIOD)
    ema_12h_slope = np.diff(ema_12h, prepend=ema_12h[0])  # slope approximation
    ema_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_slope)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    upper_channel, lower_channel = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_12h_slope_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(volume_ma[i]):
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
        
        # Breakout conditions
        breakout_up = close[i] > upper_channel[i]
        breakdown_down = close[i] < lower_channel[i]
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 12h EMA slope
        uptrend = ema_12h_slope_aligned[i] > 0
        downtrend = ema_12h_slope_aligned[i] < 0
        
        # Generate signals
        if position == 0:
            if breakout_up and volume_ok and uptrend:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakdown_down and volume_ok and downtrend:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on breakdown or stop loss
            if breakdown_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on breakout or stop loss
            if breakout_up:
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

# Hypothesis: 4h Camarilla pivot levels with 12h trend filter and volume confirmation.
# Camarilla pivots identify key support/resistance levels based on the previous day's
# price action. Price often reacts at these levels, especially L3/L3 and H3/H3 levels.
# In trending markets, price breaking through H4 or L4 with volume confirmation
# signals continuation. The 12h EMA(50) slope filters for trades in the direction
# of the higher timeframe trend. Volume ensures institutional participation.
# Works in bull markets (buy breaks above H4 in uptrend) and bear markets (sell
# breaks below L4 in downtrend). Target: 20-40 trades/year by requiring breaks
# beyond Camarilla H4/L4 levels, trend alignment, and volume >1.5x average.

name = "exp_13653_4h_camarilla_12h_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 1  # Uses previous bar's high/low/close
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the given high, low, close.
    Returns (H4, H3, L3, L4) where:
    H4 = close + 1.5 * (high - low)
    H3 = close + 1.125 * (high - low)
    L3 = close - 1.125 * (high - low)
    L4 = close - 1.5 * (high - low)
    """
    range_ = high - low
    H4 = close + 1.5 * range_
    H3 = close + 1.125 * range_
    L3 = close - 1.125 * range_
    L4 = close - 1.5 * range_
    return H4, H3, L3, L4

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, TREND_EMA_PERIOD)
    ema_12h_slope = np.diff(ema_12h, prepend=ema_12h[0])  # slope approximation
    ema_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_slope)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels using previous bar's data
    # We need to shift high, low, close by 1 to get previous bar's values
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    # First bar has no previous data, so we'll use current values (will be filtered out by warmup)
    high_prev[0] = high[0]
    low_prev[0] = low[0]
    close_prev[0] = close[0]
    
    camarilla_H4, camarilla_H3, camarilla_L3, camarilla_L4 = calculate_camarilla(high_prev, low_prev, close_prev)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_12h_slope_aligned[i]) or np.isnan(camarilla_H4[i]) or np.isnan(camarilla_L4[i]) or np.isnan(volume_ma[i]):
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
        
        # Breakout conditions using Camarilla levels
        breakout_H4 = close[i] > camarilla_H4[i]
        breakdown_L4 = close[i] < camarilla_L4[i]
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 12h EMA slope
        uptrend = ema_12h_slope_aligned[i] > 0
        downtrend = ema_12h_slope_aligned[i] < 0
        
        # Generate signals
        if position == 0:
            if breakout_H4 and volume_ok and uptrend:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakdown_L4 and volume_ok and downtrend:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on breakdown below L3 or stop loss
            if close[i] < camarilla_L3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on breakout above H3 or stop loss
            if close[i] > camarilla_H3[i]:
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

# Hypothesis: 4h Williams %R with 12h trend filter and volume confirmation.
# Williams %R is a momentum oscillator that measures overbought/oversold levels.
# Values above -20 indicate overbought, below -80 indicate oversold.
# In trending markets, pulling back to oversold (-80) in an uptrend or
# overbought (-20) in a downtrend offers high-probability entries.
# The 12h EMA(50) slope filters for trades in the direction of the higher
# timeframe trend. Volume confirmation ensures breakouts are supported.
# Works in bull markets (buy pullbacks to -80 in uptrend) and bear markets
# (sell rallies to -20 in downtrend). Target: 25-50 trades/year by requiring
# Williams %R extremes, trend alignment, and volume >1.5x average.

name = "exp_13653_4h_williamsr_12h_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
WILLIAMS_PERIOD = 14
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_williams_r(high, low, close, period):
    """Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return williams_r.values

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, TREND_EMA_PERIOD)
    ema_12h_slope = np.diff(ema_12h, prepend=ema_12h[0])  # slope approximation
    ema_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_slope)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R
    williams_r = calculate_williams_r(high, low, close, WILLIAMS_PERIOD)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WILLIAMS_PERIOD, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_12h_slope_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(volume_ma[i]):
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
        
        # Williams %R conditions
        oversold = williams_r[i] <= -80
        overbought = williams_r[i] >= -20
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 12h EMA slope
        uptrend = ema_12h_slope_aligned[i] > 0
        downtrend = ema_12h_slope_aligned[i] < 0
        
        # Generate signals
        if position == 0:
            if oversold and volume_ok and uptrend:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif overbought and volume_ok and downtrend:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when Williams %R reaches overbought or stop loss
            if williams_r[i] >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when Williams %R reaches oversold or stop loss
            if williams_r[i] <= -80:
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

# Hypothesis: 4h EFI (Elder's Force Index) with 12h trend filter and volume confirmation.
# EFI = volume * (close - prior close) measures the power behind price movements.
# It combines price and volume to assess trend strength. High positive EFI indicates
# strong buying pressure, negative indicates selling pressure. The 12h EMA(50) slope
# filters for trades in the direction of the higher timeframe trend. We look for
# EFI extremes (>20,000 or <-20,000) as signals of exhaustion, expecting mean
# reversion. Works in bull markets (sell on extreme positive EFI) and bear markets
# (buy on extreme negative EFI). Target: 20-40 trades/year by requiring EFI
# extremes, trend alignment (counter-trend to 12h), and volume confirmation.

name = "exp_13653_4h_efi_12h_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
EFI_PERIOD = 13  # Typical EFI smoothing period
TREND_EMA_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
EFI_THRESHOLD = 20000  # Absolute threshold for extreme values
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
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, TREND_EMA_PERIOD)
    ema_12h_slope = np.diff(ema_12h, prepend=ema_12h[0])  # slope approximation
    ema_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_slope)
    
    # Calculate 4h indicators
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate EFI: volume * (close - prior close)
    close_prev = np.roll(close, 1)
    close_prev[0] = close[0]  # First bar uses current close
    price_change = close - close_prev
    efi_raw = volume * price_change
    # Smooth EFI with EMA
    efi = pd.Series(efi_raw).ewm(span=EFI_PERIOD, adjust=False, min_periods=EFI_PERIOD).mean().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EFI_PERIOD, TREND_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_12h_slope_aligned[i]) or np.isnan(efi[i]) or np.isnan(volume_ma[i]):
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
        
        # EFI conditions: extreme values suggest exhaustion
        efi_extreme_pos = efi[i] > EFI_THRESHOLD
        efi_extreme_neg = efi[i] < -EFI_THRESHOLD
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend direction from 12h EMA slope
        uptrend = ema_12h_slope_aligned[i] > 0
        downtrend = ema_12h_slope_aligned[i] < 0
        
        # Generate signals: fade extreme EFI (counter-trend to 12h EMA)
        if position == 0:
            if efi_extreme_neg and volume_ok and uptrend:
                # Extreme negative EFI in uptrend = buying exhaustion, go long
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif efi_extreme_pos and volume_ok and downtrend:
                # Extreme positive EFI in downtrend = selling exhaustion, go short
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when EFI returns to neutral or stop loss
            if efi[i] < 0:  # EFI turned negative
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when EFI returns to neutral or stop loss
            if efi[i] > 0:  # EFI turned positive
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

# Hypothesis: 4h Chaikin Oscillator with 12h trend filter and volume confirmation.
# Chaikin Oscillator = EMA(3, ADL) - EMA(10, ADL) where ADL = Accumulation/Distribution Line.
# It measures the momentum of accumulation/distribution. Values above zero indicate
# buying pressure, below zero indicate selling pressure. The 12h EMA(50) slope
# filters for trades in the direction of the higher timeframe trend. We look for
# zero-line crosses with volume confirmation. Works in bull markets (buy when
# CO crosses above zero in uptrend) and bear markets (sell when CO crosses below
# zero in downtrend). Target: 25-50 trades/year by requiring zero-line crosses,
# trend alignment, and volume >1.5x average.

name = "exp_13653_4h_chaikin_12h_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
CHAIKIN_FAST = 3
CHAIKIN_SLOW = 10
TREND_EMA_PERIOD = 50
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
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, TREND_EMA_PERIOD)
    ema_12h_slope = np.diff(ema_12h, prepend=ema_12h[0])  # slope approximation
    ema_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_slope)
    
    # Calculate 4h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Accumulation/Distribution Line (ADL)
    # ADL = previous ADL + ((close - low) - (high - close)) / (high - low) * volume
    # When high == low, we use 0 to avoid division by zero
    high_low = high - low
    high_low[high_low == 0] = 1e-10  # Avoid division by zero
    clv = ((close - low) - (high - close)) / high_low  # Close Location Value
    adl = np.cumsum(clv * volume)  # Accumulation/Distribution Line
    
    # Chaikin Oscillator: EMA(3, ADL) - EMA(10, ADL)
    adl_ema_fast = pd.Series(adl).ewm(span=CHAIKIN_FAST, adjust=False, min_periods=CHAIKIN_FAST).mean().values
    adl_ema_slow = pd.Series(adl).ewm(span=CHAIKIN_SLOW, adjust=False, min_periods=CHAIKIN_SLOW).mean().values
    chaikin_osc = adl_ema_fast - adl_ema_slow
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price =