#!/usr/bin/env python3
"""
exp_7516_12h_donchian20_1d_ema_vol_v1
Hypothesis: 12h Donchian breakout with 1d EMA trend filter and volume confirmation.
Breakouts above 20-period upper band + volume surge + price above 1d EMA200 = long.
Breakouts below 20-period lower band + volume surge + price below 1d EMA200 = short.
Targets 50-150 trades over 4 years (12-37/year) with strict breakout conditions.
Uses ATR-based stoploss to limit drawdown.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7516_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_TREND = 200
VOLUME_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_1d_200 = pd.Series(close_1d).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume average
    volume_ma = pd.Series(volume).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).mean().values
    
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
    start = max(DONCHIAN_PERIOD, EMA_TREND, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1d_200_aligned[i]):
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
        
        # Determine market regime
        above_ema200 = close[i] > ema_1d_200_aligned[i]  # bull regime
        below_ema200 = close[i] < ema_1d_200_aligned[i]  # bear regime
        
        # Breakout conditions
        upper_breakout = close[i] > highest_high[i-1]  # break above previous high
        lower_breakout = close[i] < lowest_low[i-1]    # break below previous low
        
        # Volume confirmation
        volume_surge = volume[i] > VOLUME_MULTIPLIER * volume_ma[i-1]
        
        # Entry conditions
        long_entry = (
            upper_breakout and      # breakout above Donchian high
            volume_surge and        # volume confirmation
            above_ema200            # bull regime filter
        )
        
        short_entry = (
            lower_breakout and      # breakdown below Donchian low
            volume_surge and        # volume confirmation
            below_ema200            # bear regime filter
        )
        
        # Exit conditions (exit when price crosses back through the opposite band)
        long_exit = close[i] < lowest_low[i]
        short_exit = close[i] > highest_high[i]
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals
</think>
#!/usr/bin/env python3
"""
exp_7516_12h_donchian20_1d_ema_vol_v1
Hypothesis: 12h Donchian breakout with 1d EMA trend filter and volume confirmation.
Breakouts above 20-period upper band + volume surge + price above 1d EMA200 = long.
Breakouts below 20-period lower band + volume surge + price below 1d EMA200 = short.
Targets 50-150 trades over 4 years (12-37/year) with strict breakout conditions.
Uses ATR-based stoploss to limit drawdown.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7516_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_TREND = 200
VOLUME_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_1d_200 = pd.Series(close_1d).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1d_200_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_200)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume average
    volume_ma = pd.Series(volume).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).mean().values
    
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
    start = max(DONCHIAN_PERIOD, EMA_TREND, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1d_200_aligned[i]):
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
        
        # Determine market regime
        above_ema200 = close[i] > ema_1d_200_aligned[i]  # bull regime
        below_ema200 = close[i] < ema_1d_200_aligned[i]  # bear regime
        
        # Breakout conditions
        upper_breakout = close[i] > highest_high[i-1]  # break above previous high
        lower_breakout = close[i] < lowest_low[i-1]    # break below previous low
        
        # Volume confirmation
        volume_surge = volume[i] > VOLUME_MULTIPLIER * volume_ma[i-1]
        
        # Entry conditions
        long_entry = (
            upper_breakout and      # breakout above Donchian high
            volume_surge and        # volume confirmation
            above_ema200            # bull regime filter
        )
        
        short_entry = (
            lower_breakout and      # breakdown below Donchian low
            volume_surge and        # volume confirmation
            below_ema200            # bear regime filter
        )
        
        # Exit conditions (exit when price crosses back through the opposite band)
        long_exit = close[i] < lowest_low[i]
        short_exit = close[i] > highest_high[i]
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals