#!/usr/bin/env python3
"""
exp_7511_6d_1w_ichimoku_cloud_v1
Hypothesis: 6s Ichimoku Cloud with 1-week trend filter for trend-following entries.
In bull markets (price > 1w Kumo): enter long when Tenkan > Kijun and price > Kumo.
In bear markets (price < 1w Kumo): enter short when Tenkan < Kijun and price < Kumo.
Uses Kumo as dynamic support/resistance. Targets 75-150 trades over 4 years (19-38/year).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7511_6d_1w_ichimoku_cloud_v1"
timeframe = "6h"
leverage = 1.0

# Ichimoku parameters
TENKAN_PERIOD = 9   # Conversion Line
KIJUN_PERIOD = 26   # Base Line
SENKOU_B_PERIOD = 52 # Leading Span B
KUMO_SHIFT = 26     # Kumo displacement forward

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max()
    period9_low = pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max()
    period26_low = pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max()
    period52_low = pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()
    senkou_b = (period52_high + period52_low) / 2
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w Ichimoku for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w = calculate_ichimoku(high_1w, low_1w, close_1w)
    
    # Kumo (Cloud) boundaries - shifted forward by KUMO_SHIFT periods
    senkou_a_shifted = np.roll(senkou_a_1w, KUMO_SHIFT)
    senkou_b_shifted = np.roll(senkou_b_1w, KUMO_SHIFT)
    # Set first KUMO_SHIFT values to NaN (no data available)
    senkou_a_shifted[:KUMO_SHIFT] = np.nan
    senkou_b_shifted[:KUMO_SHIFT] = np.nan
    
    # Kumo top and bottom
    kumo_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    kumo_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # Align 1w Ichimoku components to 6s timeframe
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    kumo_top_aligned = align_htf_to_ltf(prices, df_1w, kumo_top)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1w, kumo_bottom)
    
    # Calculate 6s Ichimoku for entry signals
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h = calculate_ichimoku(high, low, close)
    
    # Kumo on 6s
    senkou_a_shifted_6h = np.roll(senkou_a_6h, KUMO_SHIFT)
    senkou_b_shifted_6h = np.roll(senkou_b_6h, KUMO_SHIFT)
    senkou_a_shifted_6h[:KUMO_SHIFT] = np.nan
    senkou_b_shifted_6h[:KUMO_SHIFT] = np.nan
    kumo_top_6h = np.maximum(senkou_a_shifted_6h, senkou_b_shifted_6h)
    kumo_bottom_6h = np.minimum(senkou_a_shifted_6h, senkou_b_shifted_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Parameters
    SIGNAL_SIZE = 0.25
    ATR_PERIOD = 14
    ATR_STOP_MULTIPLIER = 2.5
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Start from warmup period
    start = max(KIJUN_PERIOD, SENKOU_B_PERIOD, KUMO_SHIFT) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or \
           np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i]):
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
        
        # Determine market regime from 1w Ichimoku
        price_above_kumo = close[i] > kumo_top_aligned[i]
        price_below_kumo = close[i] < kumo_bottom_aligned[i]
        
        # 6s Ichimoku entry signals
        tenkan_above_kijun = tenkan_6h[i] > kijun_6h[i]
        tenkan_below_kijun = tenkan_6h[i] < kijun_6h[i]
        
        # Entry conditions
        long_entry = (
            price_above_kumo and      # bull regime (price above 1w Kumo)
            tenkan_above_kijun        # bullish momentum (Tenkan > Kijun)
        )
        
        short_entry = (
            price_below_kumo and      # bear regime (price below 1w Kumo)
            tenkan_below_kijun        # bearish momentum (Tenkan < Kijun)
        )
        
        # Exit conditions - reverse signals
        long_exit = tenkan_below_kijun  # exit when momentum turns bearish
        short_exit = tenkan_above_kijun  # exit when momentum turns bullish
        
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
exp_7511_6d_1w_ichimoku_cloud_v1
Hypothesis: 6s Ichimoku Cloud with 1-week trend filter for trend-following entries.
In bull markets (price > 1w Kumo): enter long when Tenkan > Kijun and price > Kumo.
In bear markets (price < 1w Kumo): enter short when Tenkan < Kijun and price < Kumo.
Uses Kumo as dynamic support/resistance. Targets 75-150 trades over 4 years (19-38/year).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7511_6d_1w_ichimoku_cloud_v1"
timeframe = "6h"
leverage = 1.0

# Ichimoku parameters
TENKAN_PERIOD = 9   # Conversion Line
KIJUN_PERIOD = 26   # Base Line
SENKOU_B_PERIOD = 52 # Leading Span B
KUMO_SHIFT = 26     # Kumo displacement forward

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max()
    period9_low = pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max()
    period26_low = pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max()
    period52_low = pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()
    senkou_b = (period52_high + period52_low) / 2
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w Ichimoku for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w = calculate_ichimoku(high_1w, low_1w, close_1w)
    
    # Kumo (Cloud) boundaries - shifted forward by KUMO_SHIFT periods
    senkou_a_shifted = np.roll(senkou_a_1w, KUMO_SHIFT)
    senkou_b_shifted = np.roll(senkou_b_1w, KUMO_SHIFT)
    # Set first KUMO_SHIFT values to NaN (no data available)
    senkou_a_shifted[:KUMO_SHIFT] = np.nan
    senkou_b_shifted[:KUMO_SHIFT] = np.nan
    
    # Kumo top and bottom
    kumo_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    kumo_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # Align 1w Ichimoku components to 6s timeframe
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    kumo_top_aligned = align_htf_to_ltf(prices, df_1w, kumo_top)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1w, kumo_bottom)
    
    # Calculate 6s Ichimoku for entry signals
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h = calculate_ichimoku(high, low, close)
    
    # Kumo on 6s
    senkou_a_shifted_6h = np.roll(senkou_a_6h, KUMO_SHIFT)
    senkou_b_shifted_6h = np.roll(senkou_b_6h, KUMO_SHIFT)
    senkou_a_shifted_6h[:KUMO_SHIFT] = np.nan
    senkou_b_shifted_6h[:KUMO_SHIFT] = np.nan
    kumo_top_6h = np.maximum(senkou_a_shifted_6h, senkou_b_shifted_6h)
    kumo_bottom_6h = np.minimum(senkou_a_shifted_6h, senkou_b_shifted_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Parameters
    SIGNAL_SIZE = 0.25
    ATR_PERIOD = 14
    ATR_STOP_MULTIPLIER = 2.5
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Start from warmup period
    start = max(KIJUN_PERIOD, SENKOU_B_PERIOD, KUMO_SHIFT) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or \
           np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i]):
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
        
        # Determine market regime from 1w Ichimoku
        price_above_kumo = close[i] > kumo_top_aligned[i]
        price_below_kumo = close[i] < kumo_bottom_aligned[i]
        
        # 6s Ichimoku entry signals
        tenkan_above_kijun = tenkan_6h[i] > kijun_6h[i]
        tenkan_below_kijun = tenkan_6h[i] < kijun_6h[i]
        
        # Entry conditions
        long_entry = (
            price_above_kumo and      # bull regime (price above 1w Kumo)
            tenkan_above_kijun        # bullish momentum (Tenkan > Kijun)
        )
        
        short_entry = (
            price_below_kumo and      # bear regime (price below 1w Kumo)
            tenkan_below_kijun        # bearish momentum (Tenkan < Kijun)
        )
        
        # Exit conditions - reverse signals
        long_exit = tenkan_below_kijun  # exit when momentum turns bearish
        short_exit = tenkan_above_kijun  # exit when momentum turns bullish
        
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