#!/usr/bin/env python3
"""
exp_7079_6h_camarilla1d_v1
Hypothesis: 6h Camarilla pivot breakout/mean reversion with 12h trend filter.
Uses daily Camarilla levels: fade at R3/S3 (mean reversion in chop), breakout continuation at R4/S4 (trend following).
12h EMA50 determines regime: above = bullish bias (favor longs), below = bearish bias (favor shorts).
Designed for low trade frequency (~12-37/year) with volume confirmation to avoid false signals.
Works in both bull and bear markets by adapting to 12h trend direction.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7079_6h_camarilla1d_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # Standard Camarilla multiplier
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 24  # ~6 days (6h bars)
EMA_PERIOD = 50

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla and 12h for EMA
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: H-L range from previous day
    range_1d = high_1d - low_1d
    # Calculate levels based on previous day's close (standard Camarilla)
    camarilla_h4 = close_1d + (range_1d * 1.1 / 2)  # R4
    camarilla_h3 = close_1d + (range_1d * 1.1 / 4)  # R3
    camarilla_l3 = close_1d - (range_1d * 1.1 / 4)  # S3
    camarilla_l4 = close_1d - (range_1d * 1.1 / 2)  # S4
    
    # Align Camarilla levels to 6h (use previous day's levels)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(20, VOL_MA_PERIOD, ATR_PERIOD, EMA_PERIOD) + 1  # Need at least 1 day for Camarilla
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_12h_aligned[i]) or np.isnan(h4_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=1).mean().iloc[i]
        vol_confirmed = volume[i] > vol_ma * VOL_BASE_THRESHOLD if not np.isnan(vol_ma) else False
        
        # Determine trend direction from 12h EMA50
        weekly_uptrend = close[i] > ema_12h_aligned[i]
        weekly_downtrend = close[i] < ema_12h_aligned[i]
        
        # Camarilla-based signals
        # Fade at R3/S3 (mean reversion) - stronger in ranging markets
        long_fade = (close[i] <= l3_aligned[i]) and vol_confirmed
        short_fade = (close[i] >= h3_aligned[i]) and vol_confirmed
        
        # Breakout continuation at R4/S4 (trend following) - stronger in trending markets
        long_breakout = (close[i] >= h4_aligned[i]) and vol_confirmed
        short_breakout = (close[i] <= l4_aligned[i]) and vol_confirmed
        
        # Combine signals with 12h trend filter
        # In uptrend: favor long breakouts and long fades (but breakouts stronger)
        # In downtrend: favor short breakouts and short fades (but breakouts stronger)
        if position == 0:
            if weekly_uptrend:
                # Bullish bias: prefer longs
                if long_breakout:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif long_fade:
                    signals[i] = SIGNAL_SIZE * 0.5  # Half position for mean reversion
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            elif weekly_downtrend:
                # Bearish bias: prefer shorts
                if short_breakout:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif short_fade:
                    signals[i] = -SIGNAL_SIZE * 0.5  # Half position for mean reversion
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            else:
                # Neutral/no clear trend: only trade strong signals
                if long_breakout:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif short_breakout:
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
exp_7079_6h_camarilla1d_v1
Hypothesis: 6h Camarilla pivot breakout/mean reversion with 12h trend filter.
Uses daily Camarilla levels: fade at R3/S3 (mean reversion in chop), breakout continuation at R4/S4 (trend following).
12h EMA50 determines regime: above = bullish bias (favor longs), below = bearish bias (favor shorts).
Designed for low trade frequency (~12-37/year) with volume confirmation to avoid false signals.
Works in both bull and bear markets by adapting to 12h trend direction.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7079_6h_camarilla1d_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # Standard Camarilla multiplier
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 24  # ~6 days (6h bars)
EMA_PERIOD = 50

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla and 12h for EMA
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: H-L range from previous day
    range_1d = high_1d - low_1d
    # Calculate levels based on previous day's close (standard Camarilla)
    camarilla_h4 = close_1d + (range_1d * 1.1 / 2)  # R4
    camarilla_h3 = close_1d + (range_1d * 1.1 / 4)  # R3
    camarilla_l3 = close_1d - (range_1d * 1.1 / 4)  # S3
    camarilla_l4 = close_1d - (range_1d * 1.1 / 2)  # S4
    
    # Align Camarilla levels to 6h (use previous day's levels)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(20, VOL_MA_PERIOD, ATR_PERIOD, EMA_PERIOD) + 1  # Need at least 1 day for Camarilla
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_12h_aligned[i]) or np.isnan(h4_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=1).mean().iloc[i]
        vol_confirmed = volume[i] > vol_ma * VOL_BASE_THRESHOLD if not np.isnan(vol_ma) else False
        
        # Determine trend direction from 12h EMA50
        weekly_uptrend = close[i] > ema_12h_aligned[i]
        weekly_downtrend = close[i] < ema_12h_aligned[i]
        
        # Camarilla-based signals
        # Fade at R3/S3 (mean reversion) - stronger in ranging markets
        long_fade = (close[i] <= l3_aligned[i]) and vol_confirmed
        short_fade = (close[i] >= h3_aligned[i]) and vol_confirmed
        
        # Breakout continuation at R4/S4 (trend following) - stronger in trending markets
        long_breakout = (close[i] >= h4_aligned[i]) and vol_confirmed
        short_breakout = (close[i] <= l4_aligned[i]) and vol_confirmed
        
        # Combine signals with 12h trend filter
        # In uptrend: favor long breakouts and long fades (but breakouts stronger)
        # In downtrend: favor short breakouts and short fades (but breakouts stronger)
        if position == 0:
            if weekly_uptrend:
                # Bullish bias: prefer longs
                if long_breakout:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif long_fade:
                    signals[i] = SIGNAL_SIZE * 0.5  # Half position for mean reversion
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            elif weekly_downtrend:
                # Bearish bias: prefer shorts
                if short_breakout:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif short_fade:
                    signals[i] = -SIGNAL_SIZE * 0.5  # Half position for mean reversion
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            else:
                # Neutral/no clear trend: only trade strong signals
                if long_breakout:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif short_breakout:
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