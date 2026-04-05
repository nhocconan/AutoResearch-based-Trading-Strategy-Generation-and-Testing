#!/usr/bin/env python3
"""
exp_7504_1d_1w_donchian20_ema_vol_v1
Hypothesis: 1d Donchian breakout with 1w EMA trend filter and volume confirmation. 
In uptrend (price > 1w EMA200): buy breakout above Donchian high with volume > 1.5x average.
In downtrend (price < 1w EMA200): sell breakout below Donchian low with volume > 1.5x average.
Uses volume confirmation to avoid false breakouts and EMA to filter trend.
Targets 30-100 trades over 4 years (7-25/year) with strict breakout conditions.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7504_1d_1w_donchian20_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_TREND = 200
VOLUME_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_1w_200 = pd.Series(close_1w).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1w_200_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_200)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donch_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donch_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume average
    vol_avg = pd.Series(volume).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).mean().values
    
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
        if np.isnan(ema_1w_200_aligned[i]):
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
        above_ema200 = close[i] > ema_1w_200_aligned[i]  # bull regime
        below_ema200 = close[i] < ema_1w_200_aligned[i]  # bear regime
        
        # Breakout conditions with volume confirmation
        high_breakout = high[i] > donch_high[i-1]  # break above previous Donchian high
        low_breakout = low[i] < donch_low[i-1]   # break below previous Donchian low
        volume_confirm = volume[i] > vol_avg[i] * VOLUME_MULTIPLIER
        
        # Entry conditions
        long_entry = (
            above_ema200 and           # bull regime
            high_breakout and          # breakout above Donchian high
            volume_confirm             # volume confirmation
        )
        
        short_entry = (
            below_ema200 and           # bear regime
            low_breakout and           # breakout below Donchian low
            volume_confirm             # volume confirmation
        )
        
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
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
exp_7504_1d_1w_donchian20_ema_vol_v1
Hypothesis: 1d Donchian breakout with 1w EMA trend filter and volume confirmation. 
In uptrend (price > 1w EMA200): buy breakout above Donchian high with volume > 1.5x average.
In downtrend (price < 1w EMA200): sell breakout below Donchian low with volume > 1.5x average.
Uses volume confirmation to avoid false breakouts and EMA to filter trend.
Targets 30-100 trades over 4 years (7-25/year) with strict breakout conditions.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7504_1d_1w_donchian20_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_TREND = 200
VOLUME_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_1w_200 = pd.Series(close_1w).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1w_200_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_200)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    donch_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donch_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume average
    vol_avg = pd.Series(volume).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).mean().values
    
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
        if np.isnan(ema_1w_200_aligned[i]):
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
        above_ema200 = close[i] > ema_1w_200_aligned[i]  # bull regime
        below_ema200 = close[i] < ema_1w_200_aligned[i]  # bear regime
        
        # Breakout conditions with volume confirmation
        high_breakout = high[i] > donch_high[i-1]  # break above previous Donchian high
        low_breakout = low[i] < donch_low[i-1]   # break below previous Donchian low
        volume_confirm = volume[i] > vol_avg[i] * VOLUME_MULTIPLIER
        
        # Entry conditions
        long_entry = (
            above_ema200 and           # bull regime
            high_breakout and          # breakout above Donchian high
            volume_confirm             # volume confirmation
        )
        
        short_entry = (
            below_ema200 and           # bear regime
            low_breakout and           # breakout below Donchian low
            volume_confirm             # volume confirmation
        )
        
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
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals