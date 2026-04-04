#!/usr/bin/env python3
"""
exp_6563_4h_donchian20_12h_ema_vol_v1
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
Uses 12h HTF for trend (more responsive than 1d in bear markets) + volume spike + ATR stoploss.
Target: 100-180 total trades over 4 years (25-45/year). Discrete sizing 0.25 balances profit and fees.
Works in bull/bear via 12h EMA50 (avoids 2022-style crashes by switching to short bias).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6563_4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50         # 12h EMA50 for trend filter
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 2.0     # Volume must be 2x 20-period average
SIGNAL_SIZE = 0.25      # 25% position size (discrete level)
ATR_PERIOD = 14
ATR_MULT = 2.5          # Stoploss at 2.5 * ATR

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for EMA50
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=EMA_PERIOD, adjust=False).mean().values
    
    # Align to LTF (4h) with shift(1) for completed bars only
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
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
    
    # ATR for stoploss calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    entry_atr = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_12h_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss for existing position
        if position == 1:  # long position
            if close[i] < entry_price - ATR_MULT * entry_atr:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] > entry_price + ATR_MULT * entry_atr:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Long conditions: price > 12h EMA50 (bullish bias) + breaks above Donchian HIGH + volume spike
        long_bias = close[i] > ema_12h_aligned[i]  # price above 12h EMA50 (bullish)
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price < 12h EMA50 (bearish bias) + breaks below Donchian LOW + volume spike
        short_bias = close[i] < ema_12h_aligned[i]  # price below 12h EMA50 (bearish)
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Enter new positions only if flat
        if position == 0:
            if long_bias and long_breakout and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                entry_atr = atr[i]
                bars_since_entry = 0
            elif short_bias and short_breakout and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                entry_atr = atr[i]
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
exp_6563_4h_donchian20_12h_ema_vol_v1
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
Uses 12h HTF for trend (more responsive than 1d in bear markets) + volume spike + ATR stoploss.
Target: 100-180 total trades over 4 years (25-45/year). Discrete sizing 0.25 balances profit and fees.
Works in bull/bear via 12h EMA50 (avoids 2022-style crashes by switching to short bias).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6563_4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_PERIOD = 50         # 12h EMA50 for trend filter
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 2.0     # Volume must be 2x 20-period average
SIGNAL_SIZE = 0.25      # 25% position size (discrete level)
ATR_PERIOD = 14
ATR_MULT = 2.5          # Stoploss at 2.5 * ATR

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for EMA50
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=EMA_PERIOD, adjust=False).mean().values
    
    # Align to LTF (4h) with shift(1) for completed bars only
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
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
    
    # ATR for stoploss calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    entry_atr = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_12h_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss for existing position
        if position == 1:  # long position
            if close[i] < entry_price - ATR_MULT * entry_atr:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] > entry_price + ATR_MULT * entry_atr:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Long conditions: price > 12h EMA50 (bullish bias) + breaks above Donchian HIGH + volume spike
        long_bias = close[i] > ema_12h_aligned[i]  # price above 12h EMA50 (bullish)
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price < 12h EMA50 (bearish bias) + breaks below Donchian LOW + volume spike
        short_bias = close[i] < ema_12h_aligned[i]  # price below 12h EMA50 (bearish)
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Enter new positions only if flat
        if position == 0:
            if long_bias and long_breakout and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                entry_atr = atr[i]
                bars_since_entry = 0
            elif short_bias and short_breakout and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                entry_atr = atr[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals

</think>