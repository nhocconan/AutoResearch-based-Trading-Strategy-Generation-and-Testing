#!/usr/bin/env python3
"""
Experiment #9931: 6h Ichimoku Cloud + 1d Kumo Twist + Volume Spike
Hypothesis: Ichimoku cloud breakouts aligned with 1d Kumo twist (Senkou Span A/B crossover) and volume confirmation provide high-probability trend continuation. Works in bull/bear by trading with the cloud direction, filtering false signals via Kumo twist and volume spikes. Targets 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "exp_9931_6h_ichimoku_cloud_kumo_twist_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
KUMO_TWIST_LOOKBACK = 26
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ichimoku(high, low):
    """Calculate Ichimoku components"""
    tenkan = (pd.Series(high).rolling(TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
              pd.Series(low).rolling(TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    kijun = (pd.Series(high).rolling(KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
             pd.Series(low).rolling(KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(KIJUN_PERIOD)
    senkou_b = ((pd.Series(high).rolling(SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                 pd.Series(low).rolling(SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(KIJUN_PERIOD)
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    return pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data ONCE for Kumo twist
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku for Kumo twist
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d)
    
    # Kumo twist: Senkou A/B crossover (twist occurs when they cross)
    kumo_twist = np.zeros_like(senkou_a_1d)
    kumo_twist[1:] = np.where(
        (senkou_a_1d[1:] > senkou_b_1d[1:]) & (senkou_a_1d[:-1] <= senkou_b_1d[:-1]), 1,  # A crosses above B
        np.where((senkou_a_1d[1:] < senkou_b_1d[1:]) & (senkou_a_1d[:-1] >= senkou_b_1d[:-1]), -1, 0)  # A crosses below B
    )
    kumo_twist_cum = np.cumsum(kumo_twist)  # Cumulative twist signal
    
    # Align Kumo twist to 6h
    kumo_twist_aligned = align_ltf_to_htf(prices, df_1d, kumo_twist_cum)
    
    # Calculate 6h Ichimoku for entry signals
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h = calculate_ichimoku(high_6h, low_6h)
    
    # Price above/below cloud
    above_cloud = (close_6h > senkou_a_6h) & (close_6h > senkou_b_6h)
    below_cloud = (close_6h < senkou_a_6h) & (close_6h < senkou_b_6h)
    
    # TK crossover
    tk_cross_up = (tenkan_6h > kijun_6h) & (tenkan_6h <= kijun_6h)  # Cross up
    tk_cross_down = (tenkan_6h < kijun_6h) & (tenkan_6h >= kijun_6h)  # Cross down
    
    # Volume spike
    volume_ma = pd.Series(volume_6h).rolling(20, min_periods=20).mean().values
    volume_spike = volume_6h > (volume_ma * VOLUME_SPIKE_MULTIPLIER)
    
    # ATR for stops
    atr = calculate_atr(high_6h, low_6h, close_6h, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    stop_price = 0.0
    
    start = max(SENKOU_B_PERIOD + KIJUN_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if Kumo twist not available
        if np.isnan(kumo_twist_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Stoploss check
        if position == 1 and close_6h[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close_6h[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry conditions: TK cross in direction of Kumo twist with volume spike
        long_entry = tk_cross_up[i] and (kumo_twist_aligned[i] > 0) and volume_spike[i] and above_cloud[i]
        short_entry = tk_cross_down[i] and (kumo_twist_aligned[i] < 0) and volume_spike[i] and below_cloud[i]
        
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close_6h[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close_6h[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

# Note: align_ltf_to_htf is actually align_htf_to_ltf in mtf_data - correcting this
# The actual function name is align_htf_to_ltf, so we need to fix the import and usage
# Let me rewrite with correct function name
</reasoning> 

Actually, let me check the mtf_data module - the correct function is align_htf_to_ltf. Let me fix the code properly. 

</think>
#!/usr/bin/env python3
"""
Experiment #9931: 6h Ichimoku Cloud + 1d Kumo Twist + Volume Spike
Hypothesis: Ichimoku cloud breakouts aligned with 1d Kumo twist (Senkou Span A/B crossover) and volume confirmation provide high-probability trend continuation. Works in bull/bear by trading with the cloud direction, filtering false signals via Kumo twist and volume spikes. Targets 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9931_6h_ichimoku_cloud_kumo_twist_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
KUMO_TWIST_LOOKBACK = 26
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ichimoku(high, low):
    """Calculate Ichimoku components"""
    tenkan = (pd.Series(high).rolling(TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
              pd.Series(low).rolling(TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    kijun = (pd.Series(high).rolling(KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
             pd.Series(low).rolling(KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(KIJUN_PERIOD)
    senkou_b = ((pd.Series(high).rolling(SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                 pd.Series(low).rolling(SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(KIJUN_PERIOD)
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    return pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data ONCE for Kumo twist
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku for Kumo twist
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d)
    
    # Kumo twist: Senkou A/B crossover (twist occurs when they cross)
    kumo_twist = np.zeros_like(senkou_a_1d)
    kumo_twist[1:] = np.where(
        (senkou_a_1d[1:] > senkou_b_1d[1:]) & (senkou_a_1d[:-1] <= senkou_b_1d[:-1]), 1,  # A crosses above B
        np.where((senkou_a_1d[1:] < senkou_b_1d[1:]) & (senkou_a_1d[:-1] >= senkou_b_1d[:-1]), -1, 0)  # A crosses below B
    )
    kumo_twist_cum = np.cumsum(kumo_twist)  # Cumulative twist signal
    
    # Align Kumo twist to 6h
    kumo_twist_aligned = align_htf_to_ltf(prices, df_1d, kumo_twist_cum)
    
    # Calculate 6h Ichimoku for entry signals
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h = calculate_ichimoku(high_6h, low_6h)
    
    # Price above/below cloud
    above_cloud = (close_6h > senkou_a_6h) & (close_6h > senkou_b_6h)
    below_cloud = (close_6h < senkou_a_6h) & (close_6h < senkou_b_6h)
    
    # TK crossover
    tk_cross_up = (tenkan_6h > kijun_6h) & (tenkan_6h.shift(1) <= kijun_6h.shift(1))  # Cross up
    tk_cross_down = (tenkan_6h < kijun_6h) & (tenkan_6h.shift(1) >= kijun_6h.shift(1))  # Cross down
    
    # Volume spike
    volume_ma = pd.Series(volume_6h).rolling(20, min_periods=20).mean().values
    volume_spike = volume_6h > (volume_ma * VOLUME_SPIKE_MULTIPLIER)
    
    # ATR for stops
    atr = calculate_atr(high_6h, low_6h, close_6h, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    stop_price = 0.0
    
    start = max(SENKOU_B_PERIOD + KIJUN_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if Kumo twist not available
        if np.isnan(kumo_twist_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Stoploss check
        if position == 1 and close_6h[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close_6h[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry conditions: TK cross in direction of Kumo twist with volume spike
        long_entry = tk_cross_up[i] and (kumo_twist_aligned[i] > 0) and volume_spike[i] and above_cloud[i]
        short_entry = tk_cross_down[i] and (kumo_twist_aligned[i] < 0) and volume_spike[i] and below_cloud[i]
        
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close_6h[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close_6h[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals