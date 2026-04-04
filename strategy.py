#!/usr/bin/env python3
"""
Experiment #3974: 1h Donchian(20) breakout + 4h EMA20 + 1d EMA50 trend filter
HYPOTHESIS: 1h Donchian breakouts aligned with 4h EMA20 (short-term) and 1d EMA50 (medium-term) 
trend capture swings with controlled frequency. Volume > 1.8x MA(20) confirms breakout strength. 
ATR(14) trailing stop (2.0x) manages risk. Session filter (08-20 UTC) reduces noise. 
Discrete sizing (0.20) minimizes fee drag. Target: 60-150 trades over 4 years (15-37/year). 
Works in bull/bear via multi-timeframe EMA alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3974_1h_donchian20_4h_ema20_1d_ema50_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h EMA20 for short-term trend ===
    df_4h = get_htf_data(prices, '4h')
    ema_4h_20 = pd.Series(df_4h['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_4h_20_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_20)
    
    # === HTF: 1d EMA50 for medium-term trend ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d_50 = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # === 1h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Session filter: 08-20 UTC (precompute hours) ===
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20, 50)  # DC lookback, vol MA, EMA50
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_4h_20_aligned[i]) or np.isnan(ema_1d_50_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter ---
        hour = hours[i]
        if hour < 8 or hour > 20:  # Outside 08-20 UTC
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian lower band (trend reversal)
                elif price < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian upper band (trend reversal)
                elif price > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) to filter noise
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Determine trend alignment: bullish if price above BOTH 4h EMA20 and 1d EMA50
            bullish = price > ema_4h_20_aligned[i] and price > ema_1d_50_aligned[i]
            # Bearish if price below BOTH 4h EMA20 and 1d EMA50
            bearish = price < ema_4h_20_aligned[i] and price < ema_1d_50_aligned[i]
            
            # Long entry: breakout above Donchian upper band in bullish regime
            long_breakout = price > highest_high[i-1] and bullish
            # Short entry: breakdown below Donchian lower band in bearish regime
            short_breakout = price < lowest_low[i-1] and bearish
            
            if long_breakout and not short_breakout:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_breakout and not long_breakout:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #3974: 1h Donchian(20) breakout + 4h EMA20 + 1d EMA50 trend filter
HYPOTHESIS: 1h Donchian breakouts aligned with 4h EMA20 (short-term) and 1d EMA50 (medium-term) 
trend capture swings with controlled frequency. Volume > 1.8x MA(20) confirms breakout strength. 
ATR(14) trailing stop (2.0x) manages risk. Session filter (08-20 UTC) reduces noise. 
Discrete sizing (0.20) minimizes fee drag. Target: 60-150 trades over 4 years (15-37/year). 
Works in bull/bear via multi-timeframe EMA alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3974_1h_donchian20_4h_ema20_1d_ema50_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h EMA20 for short-term trend ===
    df_4h = get_htf_data(prices, '4h')
    ema_4h_20 = pd.Series(df_4h['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_4h_20_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_20)
    
    # === HTF: 1d EMA50 for medium-term trend ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d_50 = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # === 1h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Session filter: 08-20 UTC (precompute hours) ===
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20, 50)  # DC lookback, vol MA, EMA50
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_4h_20_aligned[i]) or np.isnan(ema_1d_50_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter ---
        hour = hours[i]
        if hour < 8 or hour > 20:  # Outside 08-20 UTC
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian lower band (trend reversal)
                elif price < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian upper band (trend reversal)
                elif price > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) to filter noise
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Determine trend alignment: bullish if price above BOTH 4h EMA20 and 1d EMA50
            bullish = price > ema_4h_20_aligned[i] and price > ema_1d_50_aligned[i]
            # Bearish if price below BOTH 4h EMA20 and 1d EMA50
            bearish = price < ema_4h_20_aligned[i] and price < ema_1d_50_aligned[i]
            
            # Long entry: breakout above Donchian upper band in bullish regime
            long_breakout = price > highest_high[i-1] and bullish
            # Short entry: breakdown below Donchian lower band in bearish regime
            short_breakout = price < lowest_low[i-1] and bearish
            
            if long_breakout and not short_breakout:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_breakout and not long_breakout:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals