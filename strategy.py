#!/usr/bin/env python3
"""
exp_6579_6h_donchian20_1d_pivot_vol_v2
Hypothesis: 6h Donchian(20) breakout with 1d Camarilla pivot direction and volume confirmation.
Improved version: Added dynamic bias based on price position relative to Camarilla H3/L3 levels.
In bullish bias (price > H3): favor long breakouts and fade shorts.
In bearish bias (price < L3): favor short breakouts and fade longs.
Neutral zone (between H3/L3): mean reversion at extremes.
Uses 6h timeframe for optimal trade frequency (target: 75-150 total trades over 4 years).
Volume confirmation filters low-conviction breakouts.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6579_6h_donchian20_1d_pivot_vol_v2"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
MAX_HOLD_BARS = 30  # ~7.5 days max hold

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3
    # Camarilla levels
    camarilla_h4 = pivot + (high_1d - low_1d) * 1.1 / 2  # R4
    camarilla_h3 = pivot + (high_1d - low_1d) * 1.1 / 4  # R3
    camarilla_l3 = pivot - (high_1d - low_1d) * 1.1 / 4  # S3
    camarilla_l4 = pivot - (high_1d - low_1d) * 1.1 / 2  # S4
    
    # Align to LTF (6h) with shift(1) for completed bars only
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Price position relative to Camarilla levels
        price_above_h3 = close[i] > camarilla_h3_aligned[i]
        price_below_l3 = close[i] < camarilla_l3_aligned[i]
        
        # Long conditions:
        # 1. Break above Donchian HIGH (breakout)
        # 2. Volume confirmation (> 2x average volume)
        # 3. Bias filter: 
        #    - Continuation: price > H3 (bullish bias favors longs)
        #    - Fade: price < L3 but > L4 (mean reversion from extreme)
        long_breakout = close[i] > donchian_high[i-1]
        long_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        long_continuation = price_above_h3  # Bullish bias
        long_fade = price_below_l3 and close[i] > camarilla_l4_aligned[i]  # Fade from S3 but above S4
        
        # Short conditions:
        # 1. Break below Donchian LOW (breakdown)
        # 2. Volume confirmation
        # 3. Bias filter:
        #    - Continuation: price < L3 (bearish bias favors shorts)
        #    - Fade: price > H3 but < H4 (mean reversion from extreme)
        short_breakout = close[i] < donchian_low[i-1]
        short_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        short_continuation = price_below_l3  # Bearish bias
        short_fade = price_above_h3 and close[i] < camarilla_h4_aligned[i]  # Fade from H3 but below H4
        
        # Exit conditions: Donchian midpoint reversal OR time-based exit
        if position == 1:  # long position
            # Exit if price drops below Donchian midpoint
            exit_long = close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2
            # Time-based exit: prevent overstaying in losing/choppy markets
            exit_long = exit_long or bars_since_entry >= MAX_HOLD_BARS
            if exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises above Donchian midpoint
            exit_short = close[i] > (donchian_high[i-1] + donchian_low[i-1]) / 2
            # Time-based exit
            exit_short = exit_short or bars_since_entry >= MAX_HOLD_BARS
            if exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            if (long_breakout or long_fade) and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif (short_breakout or short_fade) and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
exp_6579_6h_donchian20_1d_pivot_vol_v2
Hypothesis: 6h Donchian(20) breakout with 1d Camarilla pivot direction and volume confirmation.
Improved version: Added dynamic bias based on price position relative to Camarilla H3/L3 levels.
In bullish bias (price > H3): favor long breakouts and fade shorts.
In bearish bias (price < L3): favor short breakouts and fade longs.
Neutral zone (between H3/L3): mean reversion at extremes.
Uses 6h timeframe for optimal trade frequency (target: 75-150 total trades over 4 years).
Volume confirmation filters low-conviction breakouts.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6579_6h_donchian20_1d_pivot_vol_v2"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
MAX_HOLD_BARS = 30  # ~7.5 days max hold

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3
    # Camarilla levels
    camarilla_h4 = pivot + (high_1d - low_1d) * 1.1 / 2  # R4
    camarilla_h3 = pivot + (high_1d - low_1d) * 1.1 / 4  # R3
    camarilla_l3 = pivot - (high_1d - low_1d) * 1.1 / 4  # S3
    camarilla_l4 = pivot - (high_1d - low_1d) * 1.1 / 2  # S4
    
    # Align to LTF (6h) with shift(1) for completed bars only
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Price position relative to Camarilla levels
        price_above_h3 = close[i] > camarilla_h3_aligned[i]
        price_below_l3 = close[i] < camarilla_l3_aligned[i]
        
        # Long conditions:
        # 1. Break above Donchian HIGH (breakout)
        # 2. Volume confirmation (> 2x average volume)
        # 3. Bias filter: 
        #    - Continuation: price > H3 (bullish bias favors longs)
        #    - Fade: price < L3 but > L4 (mean reversion from extreme)
        long_breakout = close[i] > donchian_high[i-1]
        long_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        long_continuation = price_above_h3  # Bullish bias
        long_fade = price_below_l3 and close[i] > camarilla_l4_aligned[i]  # Fade from S3 but above S4
        
        # Short conditions:
        # 1. Break below Donchian LOW (breakdown)
        # 2. Volume confirmation
        # 3. Bias filter:
        #    - Continuation: price < L3 (bearish bias favors shorts)
        #    - Fade: price > H3 but < H4 (mean reversion from extreme)
        short_breakout = close[i] < donchian_low[i-1]
        short_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        short_continuation = price_below_l3  # Bearish bias
        short_fade = price_above_h3 and close[i] < camarilla_h4_aligned[i]  # Fade from H3 but below H4
        
        # Exit conditions: Donchian midpoint reversal OR time-based exit
        if position == 1:  # long position
            # Exit if price drops below Donchian midpoint
            exit_long = close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2
            # Time-based exit: prevent overstaying in losing/choppy markets
            exit_long = exit_long or bars_since_entry >= MAX_HOLD_BARS
            if exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises above Donchian midpoint
            exit_short = close[i] > (donchian_high[i-1] + donchian_low[i-1]) / 2
            # Time-based exit
            exit_short = exit_short or bars_since_entry >= MAX_HOLD_BARS
            if exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            if (long_breakout or long_fade) and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif (short_breakout or short_fade) and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals