#!/usr/bin/env python3
"""
Experiment #7879: 6-hour Ichimoku Cloud with 1-day trend filter and volume confirmation.
Hypothesis: Ichimoku TK cross signals when price is above/below daily cloud, with volume >1.5x 20-period MA, capture high-probability trend continuation. Daily cloud provides major support/resistance to avoid whipsaw in ranging markets. TK cross gives timely entry while cloud filter ensures alignment with higher timeframe trend. Designed for 50-150 trades over 4 years with controlled risk.
"""

from mtf_data import get_hef_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7879_6h_ichimoku1d_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TK_PERIOD = 9
KJ_BASE_PERIOD = 26
SENKOU_B_PERIOD = 52
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou A/B, Chikou"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = (pd.Series(high).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).max() + 
              pd.Series(low).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = (pd.Series(high).rolling(window=KJ_BASE_PERIOD, min_periods=KJ_BASE_PERIOD).max() + 
             pd.Series(low).rolling(window=KJ_BASE_PERIOD, min_periods=KJ_BASE_PERIOD).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_a = (tenkan + kijun) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_b = (pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Determine cloud top/bottom and price relative to cloud
    cloud_top = np.maximum(senkou_a_1d, senkou_b_1d)
    cloud_bottom = np.minimum(senkou_a_1d, senkou_b_1d)
    price_above_cloud = close_1d > cloud_top
    price_below_cloud = close_1d < cloud_bottom
    price_in_cloud = ~(price_above_cloud | price_below_cloud)
    
    # Align to LTF
    price_above_cloud_aligned = align_htf_to_ltf(prices, df_1d, price_above_cloud)
    price_below_cloud_aligned = align_htf_to_ltf(prices, df_1d, price_below_cloud)
    price_in_cloud_aligned = align_htf_to_ltf(prices, df_1d, price_in_cloud)
    
    # TK cross signals on daily
    tk_cross_up = (tenkan_1d > kijun_1d) & (tenkan_1d <= kijun_1d)  # cross up
    tk_cross_down = (tenkan_1d < kijun_1d) & (tenkan_1d >= kijun_1d)  # cross down
    tk_cross_up_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_up)
    tk_cross_down_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_down)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(KJ_BASE_PERIOD, SENKOU_B_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_above_cloud_aligned[i]) or np.isnan(price_below_cloud_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine trend from daily cloud
        bull_trend = price_above_cloud_aligned[i]  # price above daily cloud
        bear_trend = price_below_cloud_aligned[i]  # price below daily cloud
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # TK cross signals
        tk_up = tk_cross_up_aligned[i]
        tk_down = tk_cross_down_aligned[i]
        
        # Entry conditions
        long_entry = bull_trend and tk_up and volume_confirmed
        short_entry = bear_trend and tk_down and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
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
Experiment #7879: 6-hour Ichimoku Cloud with 1-day trend filter and volume confirmation.
Hypothesis: Ichimoku TK cross signals when price is above/below daily cloud, with volume >1.5x 20-period MA, capture high-probability trend continuation. Daily cloud provides major support/resistance to avoid whipsaw in ranging markets. TK cross gives timely entry while cloud filter ensures alignment with higher timeframe trend. Designed for 50-150 trades over 4 years with controlled risk.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7879_6h_ichimoku1d_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TK_PERIOD = 9
KJ_BASE_PERIOD = 26
SENKOU_B_PERIOD = 52
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou A/B, Chikou"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = (pd.Series(high).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).max() + 
              pd.Series(low).rolling(window=TK_PERIOD, min_periods=TK_PERIOD).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = (pd.Series(high).rolling(window=KJ_BASE_PERIOD, min_periods=KJ_BASE_PERIOD).max() + 
             pd.Series(low).rolling(window=KJ_BASE_PERIOD, min_periods=KJ_BASE_PERIOD).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_a = (tenkan + kijun) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_b = (pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Determine cloud top/bottom and price relative to cloud
    cloud_top = np.maximum(senkou_a_1d, senkou_b_1d)
    cloud_bottom = np.minimum(senkou_a_1d, senkou_b_1d)
    price_above_cloud = close_1d > cloud_top
    price_below_cloud = close_1d < cloud_bottom
    price_in_cloud = ~(price_above_cloud | price_below_cloud)
    
    # Align to LTF
    price_above_cloud_aligned = align_htf_to_ltf(prices, df_1d, price_above_cloud)
    price_below_cloud_aligned = align_htf_to_ltf(prices, df_1d, price_below_cloud)
    price_in_cloud_aligned = align_htf_to_ltf(prices, df_1d, price_in_cloud)
    
    # TK cross signals on daily
    tk_cross_up = (tenkan_1d > kijun_1d) & (tenkan_1d <= kijun_1d)  # cross up
    tk_cross_down = (tenkan_1d < kijun_1d) & (tenkan_1d >= kijun_1d)  # cross down
    tk_cross_up_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_up)
    tk_cross_down_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_down)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(KJ_BASE_PERIOD, SENKOU_B_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_above_cloud_aligned[i]) or np.isnan(price_below_cloud_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine trend from daily cloud
        bull_trend = price_above_cloud_aligned[i]  # price above daily cloud
        bear_trend = price_below_cloud_aligned[i]  # price below daily cloud
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # TK cross signals
        tk_up = tk_cross_up_aligned[i]
        tk_down = tk_cross_down_aligned[i]
        
        # Entry conditions
        long_entry = bull_trend and tk_up and volume_confirmed
        short_entry = bear_trend and tk_down and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals