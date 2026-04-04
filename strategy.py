#!/usr/bin/env python3
"""
Experiment #4715: 6h Ichimoku Cloud + Volume Spike Reversal
HYPOTHESIS: On 6h timeframe, Ichimoku cloud acts as dynamic support/resistance. When price rejects the cloud with volume spike (>2x average), it indicates institutional defense of key levels, leading to mean reversion. Works in both bull/bear markets as cloud adapts to volatility and trend. Target: 12-37 trades/year via strict volume confirmation + cloud rejection logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4715_6h_ichimoku_cloud_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Ichimoku Cloud ===
    if len(df_1d) >= 52:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        # Tenkan-sen (Conversion Line): (9-period high + low)/2
        period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
        period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
        tenkan = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + low)/2
        period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
        period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
        kijun = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
        senkou_a = ((tenkan + kijun) / 2)
        
        # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
        period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
        period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
        senkou_b = ((period52_high + period52_low) / 2)
        
        # Align to 6h timeframe (shifted by 1 for completed daily bar)
        tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
        kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
        senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
        senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    else:
        tenkan_aligned = np.full(n, np.nan)
        kijun_aligned = np.full(n, np.nan)
        senkou_a_aligned = np.full(n, np.nan)
        senkou_b_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume confirmation ===
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
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(26, 20, 14)  # Ichimoku base line, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Determine cloud boundaries (Senkou Span A/B)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
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
        # Volume filter: strong confirmation (>2.0x average)
        vol_confirm = vol_ratio[i] > 2.0
        
        # Cloud rejection signals with volume confirmation
        # Long: price rejects cloud bottom from below with volume spike
        cloud_reject_long = (price > cloud_bottom) and (close[i-1] <= cloud_bottom) and vol_confirm
        # Short: price rejects cloud top from above with volume spike
        cloud_reject_short = (price < cloud_top) and (close[i-1] >= cloud_top) and vol_confirm
        
        # Final entry conditions
        if cloud_reject_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif cloud_reject_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #4715: 6h Ichimoku Cloud + Volume Spike Reversal
HYPOTHESIS: On 6h timeframe, Ichimoku cloud acts as dynamic support/resistance. When price rejects the cloud with volume spike (>2x average), it indicates institutional defense of key levels, leading to mean reversion. Works in both bull/bear markets as cloud adapts to volatility and trend. Target: 12-37 trades/year via strict volume confirmation + cloud rejection logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4715_6h_ichimoku_cloud_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Ichimoku Cloud ===
    if len(df_1d) >= 52:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        # Tenkan-sen (Conversion Line): (9-period high + low)/2
        period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
        period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
        tenkan = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + low)/2
        period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
        period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
        kijun = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
        senkou_a = ((tenkan + kijun) / 2)
        
        # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
        period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
        period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
        senkou_b = ((period52_high + period52_low) / 2)
        
        # Align to 6h timeframe (shifted by 1 for completed daily bar)
        tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
        kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
        senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
        senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    else:
        tenkan_aligned = np.full(n, np.nan)
        kijun_aligned = np.full(n, np.nan)
        senkou_a_aligned = np.full(n, np.nan)
        senkou_b_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume confirmation ===
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
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(26, 20, 14)  # Ichimoku base line, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Determine cloud boundaries (Senkou Span A/B)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
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
        # Volume filter: strong confirmation (>2.0x average)
        vol_confirm = vol_ratio[i] > 2.0
        
        # Cloud rejection signals with volume confirmation
        # Long: price rejects cloud bottom from below with volume spike
        cloud_reject_long = (price > cloud_bottom) and (close[i-1] <= cloud_bottom) and vol_confirm
        # Short: price rejects cloud top from above with volume spike
        cloud_reject_short = (price < cloud_top) and (close[i-1] >= cloud_top) and vol_confirm
        
        # Final entry conditions
        if cloud_reject_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif cloud_reject_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals