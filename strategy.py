#!/usr/bin/env python3
"""
Experiment #8391: 6-hour Camarilla Pivot Breakout with 1-day Trend Filter and Volume Confirmation.
Hypothesis: Price breaking above/below Camarilla pivot resistance/support levels (R4/S4) on 6h with volume >2x 20-period MA 
and aligned daily trend (price above/below 1-day EMA50) captures sustained moves while avoiding whipsaw. 
The daily timeframe provides longer-term context, reducing false breakouts during consolidation. 
Targeting 100-200 total trades over 4 years for optimal balance.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8391_6h_camarilla_pivot_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 20  # Use previous day's high/low/close for pivot
EMA_TREND_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    pivot = (high + low + close) / 3
    range_val = high - low
    r4 = pivot + (range_val * 1.1 / 2)
    s4 = pivot - (range_val * 1.1 / 2)
    return r4, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    
    # Price relative to 1d EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_1d > ema_1d, 1, 
                     np.where(close_1d < ema_1d, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate daily pivot levels (using previous day's high/low/close)
    # For each 1d bar, calculate pivots from that day's HLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r4 = np.zeros(len(high_1d))
    camarilla_s4 = np.zeros(len(high_1d))
    for i in range(len(high_1d)):
        r4, s4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d_vals[i])
        camarilla_r4[i] = r4
        camarilla_s4[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
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
    start = max(EMA_TREND_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]):
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
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d price above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d price below EMA50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume[i-20] if i >= 20 else 0) * VOLUME_THRESHOLD if i >= 20 else False
        
        # Breakout conditions - require close beyond Camarilla levels
        breakout_up = close[i] > camarilla_r4_aligned[i-1] if i-1 >= 0 else False
        breakout_down = close[i] < camarilla_s4_aligned[i-1] if i-1 >= 0 else False
        
        # Entry conditions
        long_entry = bull_bias and breakout_up and volume_confirmed
        short_entry = bear_bias and breakout_down and volume_confirmed
        
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
Experiment #8391: 6-hour Camarilla Pivot Breakout with 1-day Trend Filter and Volume Confirmation.
Hypothesis: Price breaking above/below Camarilla pivot resistance/support levels (R4/S4) on 6h with volume >2x 20-period MA 
and aligned daily trend (price above/below 1-day EMA50) captures sustained moves while avoiding whipsaw. 
The daily timeframe provides longer-term context, reducing false breakouts during consolidation. 
Targeting 100-200 total trades over 4 years for optimal balance.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8391_6h_camarilla_pivot_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 20  # Use previous day's high/low/close for pivot
EMA_TREND_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    pivot = (high + low + close) / 3
    range_val = high - low
    r4 = pivot + (range_val * 1.1 / 2)
    s4 = pivot - (range_val * 1.1 / 2)
    return r4, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    
    # Price relative to 1d EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_1d > ema_1d, 1, 
                     np.where(close_1d < ema_1d, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate daily pivot levels (using previous day's high/low/close)
    # For each 1d bar, calculate pivots from that day's HLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r4 = np.zeros(len(high_1d))
    camarilla_s4 = np.zeros(len(high_1d))
    for i in range(len(high_1d)):
        r4, s4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d_vals[i])
        camarilla_r4[i] = r4
        camarilla_s4[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
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
    start = max(EMA_TREND_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]):
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
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d price above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d price below EMA50
        
        # Volume confirmation - use rolling mean for stability
        volume_mean = np.mean(volume[max(0, i-VOLUME_MA_PERIOD):i]) if i >= VOLUME_MA_PERIOD else 0
        volume_confirmed = volume[i] > (volume_mean * VOLUME_THRESHOLD) if volume_mean > 0 else False
        
        # Breakout conditions - require close beyond Camarilla levels
        breakout_up = close[i] > camarilla_r4_aligned[i-1] if i-1 >= 0 else False
        breakout_down = close[i] < camarilla_s4_aligned[i-1] if i-1 >= 0 else False
        
        # Entry conditions
        long_entry = bull_bias and breakout_up and volume_confirmed
        short_entry = bear_bias and breakout_down and volume_confirmed
        
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
Experiment #8391: 6-hour Camarilla Pivot Breakout with 1-day Trend Filter and Volume Confirmation.
Hypothesis: Price breaking above/below Camarilla pivot resistance/support levels (R4/S4) on 6h with volume >2x 20-period MA 
and aligned daily trend (price above/below 1-day EMA50) captures sustained moves while avoiding whipsaw. 
The daily timeframe provides longer-term context, reducing false breakouts during consolidation. 
Targeting 100-200 total trades over 4 years for optimal balance.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8391_6h_camarilla_pivot_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 20  # Use previous day's high/low/close for pivot
EMA_TREND_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    pivot = (high + low + close) / 3
    range_val = high - low
    r4 = pivot + (range_val * 1.1 / 2)
    s4 = pivot - (range_val * 1.1 / 2)
    return r4, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    
    # Price relative to 1d EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_1d > ema_1d, 1, 
                     np.where(close_1d < ema_1d, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate daily pivot levels (using previous day's high/low/close)
    # For each 1d bar, calculate pivots from that day's HLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r4 = np.zeros(len(high_1d))
    camarilla_s4 = np.zeros(len(high_1d))
    for i in range(len(high_1d)):
        r4, s4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d_vals[i])
        camarilla_r4[i] = r4
        camarilla_s4[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
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
    start = max(EMA_TREND_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]):
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
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d price above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d price below EMA50
        
        # Volume confirmation - use rolling mean for stability
        volume_mean = np.mean(volume[max(0, i-VOLUME_MA_PERIOD):i]) if i >= VOLUME_MA_PERIOD else 0
        volume_confirmed = volume[i] > (volume_mean * VOLUME_THRESHOLD) if volume_mean > 0 else False
        
        # Breakout conditions - require close beyond Camarilla levels
        breakout_up = close[i] > camarilla_r4_aligned[i-1] if i-1 >= 0 else False
        breakout_down = close[i] < camarilla_s4_aligned[i-1] if i-1 >= 0 else False
        
        # Entry conditions
        long_entry = bull_bias and breakout_up and volume_confirmed
        short_entry = bear_bias and breakout_down and volume_confirmed
        
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
Experiment #8391: 6-hour Camarilla Pivot Breakout with 1-day Trend Filter and Volume Confirmation.
Hypothesis: Price breaking above/below Camarilla pivot resistance/support levels (R4/S4) on 6h with volume >2x 20-period MA 
and aligned daily trend (price above/below 1-day EMA50) captures sustained moves while avoiding whipsaw. 
The daily timeframe provides longer-term context, reducing false breakouts during consolidation. 
Targeting 100-200 total trades over 4 years for optimal balance.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8391_6h_camarilla_pivot_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_PERIOD = 20  # Use previous day's high/low/close for pivot
EMA_TREND_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    pivot = (high + low + close) / 3
    range_val = high - low
    r4 = pivot + (range_val * 1.1 / 2)
    s4 = pivot - (range_val * 1.1 / 2)
    return r4, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    
    # Price relative to 1d EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_1d > ema_1d, 1, 
                     np.where(close_1d < ema_1d, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate daily pivot levels (using previous day's high/low/close)
    # For each 1d bar, calculate pivots from that day's HLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r4 = np.zeros(len(high_1d))
    camarilla_s4 = np.zeros(len(high_1d))
    for i in range(len(high_1d)):
        r4, s4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d_vals[i])
        camarilla_r4[i] = r4
        camarilla_s4[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
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
    start = max(EMA_TREND_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]):
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
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d price above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d price below EMA50
        
        # Volume confirmation - use rolling mean for stability
        volume_mean = np.mean(volume[max(0, i-VOLUME_MA_PERIOD):i]) if i >= VOLUME_MA_PERIOD else 0
        volume_confirmed = volume[i] > (volume_mean * VOLUME_THRESHOLD) if volume_mean > 0 else False
        
        # Breakout conditions - require close beyond Camarilla levels
        breakout_up = close[i] > camarilla_r4_aligned[i-1] if i-1 >= 0 else False
        breakout_down = close[i] < camarilla_s4_aligned[i-1] if i-1 >= 0 else False
        
        # Entry conditions
        long_entry = bull_bias and breakout_up and volume_confirmed
        short_entry = bear_bias and breakout_down and volume_confirmed
        
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

--- END FILE ---