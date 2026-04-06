#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12791_6d_ichimoku_1d"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_SPAN_B_PERIOD = 52
SENKOU_SHIFT = 26
CHIKOU_SHIFT = 26
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
              pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
             pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_b = (pd.Series(high).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).max() + 
                pd.Series(low).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).min()) / 2
    # Chikou Span (Lagging Span): close shifted back by 26 periods
    chikou = pd.Series(close).shift(CHIKOU_SHIFT)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values, chikou.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Ichimoku
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    chikou_1d_aligned = align_htf_to_ltf(prices, df_1d, chikou_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(SENKOU_SPAN_B_PERIOD, SENKOU_SHIFT, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Ichimoku components not available
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or 
            np.isnan(chikou_1d_aligned[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Ichimoku signals
        # TK Cross: Tenkan crosses above/below Kijun
        tk_cross_up = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tk_cross_down = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        
        # Price above/below cloud
        price_above_cloud = close[i] > max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        price_below_cloud = close[i] < min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Chikou confirmation: Chikou Span above/below price from 26 periods ago
        chikou_above_price = chikou_1d_aligned[i] > close[i]
        chikou_below_price = chikou_1d_aligned[i] < close[i]
        
        # Generate signals
        if position == 0:
            # Long: TK cross up + price above cloud + Chikou above price + volume
            if tk_cross_up and price_above_cloud and chikou_above_price and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: TK cross down + price below cloud + Chikou below price + volume
            elif tk_cross_down and price_below_cloud and chikou_below_price and volume_ok:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: TK cross down or price below cloud
            if tk_cross_down or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short: TK cross up or price above cloud
            if tk_cross_up or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12791_6h_ichimoku_1d"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_SPAN_B_PERIOD = 52
SENKOU_SHIFT = 26
CHIKOU_SHIFT = 26
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
              pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
             pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_b = (pd.Series(high).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).max() + 
                pd.Series(low).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).min()) / 2
    # Chikou Span (Lagging Span): close shifted back by 26 periods
    chikou = pd.Series(close).shift(CHIKOU_SHIFT)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values, chikou.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Ichimoku
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    chikou_1d_aligned = align_htf_to_ltf(prices, df_1d, chikou_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(SENKOU_SPAN_B_PERIOD, SENKOU_SHIFT, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Ichimoku components not available
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or 
            np.isnan(chikou_1d_aligned[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Ichimoku signals
        # TK Cross: Tenkan crosses above/below Kijun
        tk_cross_up = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tk_cross_down = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        
        # Price above/below cloud
        price_above_cloud = close[i] > max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        price_below_cloud = close[i] < min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Chikou confirmation: Chikou Span above/below price from 26 periods ago
        chikou_above_price = chikou_1d_aligned[i] > close[i]
        chikou_below_price = chikou_1d_aligned[i] < close[i]
        
        # Generate signals
        if position == 0:
            # Long: TK cross up + price above cloud + Chikou above price + volume
            if tk_cross_up and price_above_cloud and chikou_above_price and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: TK cross down + price below cloud + Chikou below price + volume
            elif tk_cross_down and price_below_cloud and chikou_below_price and volume_ok:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: TK cross down or price below cloud
            if tk_cross_down or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short: TK cross up or price above cloud
            if tk_cross_up or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12791_6h_ichimoku_1d"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_SPAN_B_PERIOD = 52
SENKOU_SHIFT = 26
CHIKOU_SHIFT = 26
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
              pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
             pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_b = (pd.Series(high).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).max() + 
                pd.Series(low).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).min()) / 2
    # Chikou Span (Lagging Span): close shifted back by 26 periods
    chikou = pd.Series(close).shift(CHIKOU_SHIFT)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values, chikou.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Ichimoku
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    chikou_1d_aligned = align_htf_to_ltf(prices, df_1d, chikou_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(SENKOU_SPAN_B_PERIOD, SENKOU_SHIFT, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Ichimoku components not available
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or 
            np.isnan(chikou_1d_aligned[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Ichimoku signals
        # TK Cross: Tenkan crosses above/below Kijun
        tk_cross_up = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tk_cross_down = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        
        # Price above/below cloud
        price_above_cloud = close[i] > max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        price_below_cloud = close[i] < min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Chikou confirmation: Chikou Span above/below price from 26 periods ago
        chikou_above_price = chikou_1d_aligned[i] > close[i]
        chikou_below_price = chikou_1d_aligned[i] < close[i]
        
        # Generate signals
        if position == 0:
            # Long: TK cross up + price above cloud + Chikou above price + volume
            if tk_cross_up and price_above_cloud and chikou_above_price and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: TK cross down + price below cloud + Chikou below price + volume
            elif tk_cross_down and price_below_cloud and chikou_below_price and volume_ok:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: TK cross down or price below cloud
            if tk_cross_down or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short: TK cross up or price above cloud
            if tk_cross_up or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12791_6h_ichimoku_1d"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_SPAN_B_PERIOD = 52
SENKOU_SHIFT = 26
CHIKOU_SHIFT = 26
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
              pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
             pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_b = (pd.Series(high).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).max() + 
                pd.Series(low).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).min()) / 2
    # Chikou Span (Lagging Span): close shifted back by 26 periods
    chikou = pd.Series(close).shift(CHIKOU_SHIFT)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values, chikou.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Ichimoku
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    chikou_1d_aligned = align_htf_to_ltf(prices, df_1d, chikou_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(SENKOU_SPAN_B_PERIOD, SENKOU_SHIFT, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Ichimoku components not available
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or 
            np.isnan(chikou_1d_aligned[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Ichimoku signals
        # TK Cross: Tenkan crosses above/below Kijun
        tk_cross_up = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tk_cross_down = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        
        # Price above/below cloud
        price_above_cloud = close[i] > max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        price_below_cloud = close[i] < min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Chikou confirmation: Chikou Span above/below price from 26 periods ago
        chikou_above_price = chikou_1d_aligned[i] > close[i]
        chikou_below_price = chikou_1d_aligned[i] < close[i]
        
        # Generate signals
        if position == 0:
            # Long: TK cross up + price above cloud + Chikou above price + volume
            if tk_cross_up and price_above_cloud and chikou_above_price and volume_ok:
                signals[i] =