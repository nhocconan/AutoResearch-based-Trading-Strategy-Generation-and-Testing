#!/usr/bin/env python3
"""
exp_7211_6h_ichimoku_cloud_regime_v1
Hypothesis: 6h Ichimoku cloud with 1d trend filter. In bull regime (price > 1d Senkou Span B), 
long when price breaks above 6h Tenkan-Sen with volume confirmation. In bear regime (price < 1d Senkou Span B),
short when price breaks below 6h Tenkan-Sen with volume confirmation. Uses cloud as dynamic support/resistance.
Designed for 6h timeframe to capture swings with ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by adapting to 1d Ichomoku cloud regime.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7211_6h_ichimoku_cloud_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_PERIOD = 52
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8  # ~2 days

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Ichomoku trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_1d = (pd.Series(high_1d).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).mean() +
                 pd.Series(low_1d).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).mean()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_1d = (pd.Series(high_1d).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).mean() +
                pd.Series(low_1d).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).mean()) / 2
    
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_b_1d = (pd.Series(high_1d).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).mean() +
                   pd.Series(low_1d).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).mean()) / 2
    
    # Align to LTF (6h)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d.values)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d.values)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d.values)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d.values)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h Tenkan-sen (Conversion Line)
    tenkan_6h = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).mean() +
                 pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).mean()) / 2
    
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
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(senkou_b_1d_aligned[i]):
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
        vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().iloc[i]
        vol_confirmed = volume[i] > vol_ma * VOL_BASE_THRESHOLD if not np.isnan(vol_ma) else False
        
        # Determine market regime based on 1d Senkou Span B (cloud)
        above_cloud = close[i] > senkou_b_1d_aligned[i] and close[i] > senkou_a_1d_aligned[i]
        below_cloud = close[i] < senkou_b_1d_aligned[i] and close[i] < senkou_a_1d_aligned[i]
        in_cloud = not (above_cloud or below_cloud)
        
        # Breakout signals with volume confirmation
        tenkan_breakout_long = close[i] > tenkan_6h[i] and close[i-1] <= tenkan_6h[i-1]
        tenkan_breakout_short = close[i] < tenkan_6h[i] and close[i-1] >= tenkan_6h[i-1]
        
        # Enter new positions only if flat
        if position == 0:
            if above_cloud and tenkan_breakout_long and vol_confirmed:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif below_cloud and tenkan_breakout_short and vol_confirmed:
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
exp_7211_6h_ichimoku_cloud_regime_v1
Hypothesis: 6h Ichimoku cloud with 1d trend filter. In bull regime (price > 1d Senkou Span B), 
long when price breaks above 6h Tenkan-Sen with volume confirmation. In bear regime (price < 1d Senkou Span B),
short when price breaks below 6h Tenkan-Sen with volume confirmation. Uses cloud as dynamic support/resistance.
Designed for 6h timeframe to capture swings with ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by adapting to 1d Ichomoku cloud regime.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7211_6h_ichimoku_cloud_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_PERIOD = 52
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8  # ~2 days

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Ichomoku trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_1d = (pd.Series(high_1d).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).mean() +
                 pd.Series(low_1d).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).mean()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_1d = (pd.Series(high_1d).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).mean() +
                pd.Series(low_1d).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).mean()) / 2
    
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_b_1d = (pd.Series(high_1d).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).mean() +
                   pd.Series(low_1d).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).mean()) / 2
    
    # Align to LTF (6h)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d.values)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d.values)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d.values)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d.values)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h Tenkan-sen (Conversion Line)
    tenkan_6h = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).mean() +
                 pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).mean()) / 2
    
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
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(senkou_b_1d_aligned[i]):
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
        vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().iloc[i]
        vol_confirmed = volume[i] > vol_ma * VOL_BASE_THRESHOLD if not np.isnan(vol_ma) else False
        
        # Determine market regime based on 1d Senkou Span B (cloud)
        above_cloud = close[i] > senkou_b_1d_aligned[i] and close[i] > senkou_a_1d_aligned[i]
        below_cloud = close[i] < senkou_b_1d_aligned[i] and close[i] < senkou_a_1d_aligned[i]
        in_cloud = not (above_cloud or below_cloud)
        
        # Breakout signals with volume confirmation
        tenkan_breakout_long = close[i] > tenkan_6h[i] and close[i-1] <= tenkan_6h[i-1]
        tenkan_breakout_short = close[i] < tenkan_6h[i] and close[i-1] >= tenkan_6h[i-1]
        
        # Enter new positions only if flat
        if position == 0:
            if above_cloud and tenkan_breakout_long and vol_confirmed:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif below_cloud and tenkan_breakout_short and vol_confirmed:
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