#!/usr/bin/env python3
"""
exp_7501_4h_donchian20_1d_vol_ema_v1
Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and EMA trend filter.
Long when price breaks above Donchian upper + volume > 1.5x 10-period average + price > 1d EMA50.
Short when price breaks below Donchian lower + volume > 1.5x 10-period average + price < 1d EMA50.
Uses ATR-based stoploss (2x ATR). Targets 75-200 trades over 4 years.
Works in bull markets via breakouts and bear markets via breakdowns with volume confirmation.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7501_4h_donchian20_1d_vol_ema_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MULTIPLIER = 1.5
VOLUME_AVG_PERIOD = 10
EMA_TREND = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.25

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
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume average
    volume_avg = pd.Series(volume).rolling(window=VOLUME_AVG_PERIOD, min_periods=VOLUME_AVG_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_AVG_PERIOD, EMA_TREND, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1d_50_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_confirm = volume[i] > VOLUME_MULTIPLIER * volume_avg[i]
        
        # Breakout conditions
        bullish_breakout = (
            close[i] > highest_high[i-1] and  # break above previous Donchian high
            volume_confirm and                # volume confirmation
            close[i] > ema_1d_50_aligned[i]   # price above 1d EMA50
        )
        
        bearish_breakout = (
            close[i] < lowest_low[i-1] and    # break below previous Donchian low
            volume_confirm and                # volume confirmation
            close[i] < ema_1d_50_aligned[i]   # price below 1d EMA50
        )
        
        # Generate signals
        if position == 0:
            if bullish_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif bearish_breakout:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when price breaks below Donchian lower
            if close[i] < lowest_low[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when price breaks above Donchian upper
            if close[i] > highest_high[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
exp_7501_4h_donchian20_1d_vol_ema_v1
Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and EMA trend filter.
Long when price breaks above Donchian upper + volume > 1.5x 10-period average + price > 1d EMA50.
Short when price breaks below Donchian lower + volume > 1.5x 10-period average + price < 1d EMA50.
Uses ATR-based stoploss (2x ATR). Targets 75-200 trades over 4 years.
Works in bull markets via breakouts and bear markets via breakdowns with volume confirmation.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7501_4h_donchian20_1d_vol_ema_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MULTIPLIER = 1.5
VOLUME_AVG_PERIOD = 10
EMA_TREND = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.25

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
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume average
    volume_avg = pd.Series(volume).rolling(window=VOLUME_AVG_PERIOD, min_periods=VOLUME_AVG_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_AVG_PERIOD, EMA_TREND, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1d_50_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_confirm = volume[i] > VOLUME_MULTIPLIER * volume_avg[i]
        
        # Breakout conditions
        bullish_breakout = (
            close[i] > highest_high[i-1] and  # break above previous Donchian high
            volume_confirm and                # volume confirmation
            close[i] > ema_1d_50_aligned[i]   # price above 1d EMA50
        )
        
        bearish_breakout = (
            close[i] < lowest_low[i-1] and    # break below previous Donchian low
            volume_confirm and                # volume confirmation
            close[i] < ema_1d_50_aligned[i]   # price below 1d EMA50
        )
        
        # Generate signals
        if position == 0:
            if bullish_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif bearish_breakout:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when price breaks below Donchian lower
            if close[i] < lowest_low[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short when price breaks above Donchian upper
            if close[i] > highest_high[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>