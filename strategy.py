#!/usr/bin/env python3
"""
Experiment #4511: 6h Ichimoku Cloud Filter + 1d TK Cross + Volume Spike
HYPOTHESIS: Ichimoku cloud acts as dynamic support/resistance on 6h, while 1d TK cross (Tenkan/Kijun) provides higher-timeframe momentum confirmation. Volume spike (>2.0x average) ensures breakout conviction. Only long when price above cloud + bullish TK cross, short when price below cloud + bearish TK cross. Ichimoku's multi-line system reduces false signals in ranging markets, while TK cross captures medium-term trends. Designed for 6h timeframe targeting 75-150 total trades over 4 years (19-38/year) with position size 0.25. Works in bull/bear via cloud filter (avoids counter-trend trades) and TK cross directionality.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4511_6h_ichimoku_1d_tk_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for TK cross
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 26:  # Need for Kijun (26-period)
        high_1d = pd.Series(df_1d['high'].values)
        low_1d = pd.Series(df_1d['low'].values)
        close_1d = pd.Series(df_1d['close'].values)
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        tenkan_1d = (high_1d.rolling(window=9, min_periods=9).max() + 
                     low_1d.rolling(window=9, min_periods=9).min()) / 2
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        kijun_1d = (high_1d.rolling(window=26, min_periods=26).max() + 
                    low_1d.rolling(window=26, min_periods=26).min()) / 2
        
        # TK cross: Tenkan > Kijun = bullish, Tenkan < Kijun = bearish
        tk_bullish = (tenkan_1d > kijun_1d).values
        tk_bearish = (tenkan_1d < kijun_1d).values
        
        # Align to LTF with shift(1) to avoid look-ahead
        tk_bullish_aligned = align_htf_to_ltf(prices, df_1d, tk_bullish.astype(np.float64))
        tk_bearish_aligned = align_htf_to_ltf(prices, df_1d, tk_bearish.astype(np.float64))
    else:
        tk_bullish_aligned = np.zeros(n)
        tk_bearish_aligned = np.zeros(n)
    
    # === 6h Indicators: Ichimoku Cloud ===
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low)/2
    tenkan_6h = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    # Base Line (Kijun-sen): (26-period high + 26-period low)/2
    kijun_6h = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    # Leading Span A (Senkou Span A): (Conversion + Base)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_6h + kijun_6h) / 2).shift(26)
    # Leading Span B (Senkou Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    # Cloud: between Senkou Span A and B
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(52, 26, 20, 14)  # Ichimoku needs 52 for Senkou B
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume spike confirmation (> 2.0x average)
        volume_confirm = vol_ratio[i] > 2.0
        
        # Price relative to cloud
        price_above_cloud = price > cloud_top[i]
        price_below_cloud = price < cloud_bottom[i]
        
        # Ichimoku + TK cross conditions
        long_entry = price_above_cloud and tk_bullish_aligned[i] > 0.5 and volume_confirm
        short_entry = price_below_cloud and tk_bearish_aligned[i] > 0.5 and volume_confirm
        
        if long_entry:
            in_position = True
            position_side = 1
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #4511: 6h Ichimoku Cloud Filter + 1d TK Cross + Volume Spike
HYPOTHESIS: Ichimoku cloud acts as dynamic support/resistance on 6h, while 1d TK cross (Tenkan/Kijun) provides higher-timeframe momentum confirmation. Volume spike (>2.0x average) ensures breakout conviction. Only long when price above cloud + bullish TK cross, short when price below cloud + bearish TK cross. Ichimoku's multi-line system reduces false signals in ranging markets, while TK cross captures medium-term trends. Designed for 6h timeframe targeting 75-150 total trades over 4 years (19-38/year) with position size 0.25. Works in bull/bear via cloud filter (avoids counter-trend trades) and TK cross directionality.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4511_6h_ichimoku_1d_tk_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for TK cross
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 26:  # Need for Kijun (26-period)
        high_1d = pd.Series(df_1d['high'].values)
        low_1d = pd.Series(df_1d['low'].values)
        close_1d = pd.Series(df_1d['close'].values)
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        tenkan_1d = (high_1d.rolling(window=9, min_periods=9).max() + 
                     low_1d.rolling(window=9, min_periods=9).min()) / 2
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        kijun_1d = (high_1d.rolling(window=26, min_periods=26).max() + 
                    low_1d.rolling(window=26, min_periods=26).min()) / 2
        
        # TK cross: Tenkan > Kijun = bullish, Tenkan < Kijun = bearish
        tk_bullish = (tenkan_1d > kijun_1d).values
        tk_bearish = (tenkan_1d < kijun_1d).values
        
        # Align to LTF with shift(1) to avoid look-ahead
        tk_bullish_aligned = align_htf_to_ltf(prices, df_1d, tk_bullish.astype(np.float64))
        tk_bearish_aligned = align_htf_to_ltf(prices, df_1d, tk_bearish.astype(np.float64))
    else:
        tk_bullish_aligned = np.zeros(n)
        tk_bearish_aligned = np.zeros(n)
    
    # === 6h Indicators: Ichimoku Cloud ===
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low)/2
    tenkan_6h = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    # Base Line (Kijun-sen): (26-period high + 26-period low)/2
    kijun_6h = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    # Leading Span A (Senkou Span A): (Conversion + Base)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_6h + kijun_6h) / 2).shift(26)
    # Leading Span B (Senkou Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    # Cloud: between Senkou Span A and B
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(52, 26, 20, 14)  # Ichimoku needs 52 for Senkou B
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume spike confirmation (> 2.0x average)
        volume_confirm = vol_ratio[i] > 2.0
        
        # Price relative to cloud
        price_above_cloud = price > cloud_top[i]
        price_below_cloud = price < cloud_bottom[i]
        
        # Ichimoku + TK cross conditions
        long_entry = price_above_cloud and tk_bullish_aligned[i] > 0.5 and volume_confirm
        short_entry = price_below_cloud and tk_bearish_aligned[i] > 0.5 and volume_confirm
        
        if long_entry:
            in_position = True
            position_side = 1
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals