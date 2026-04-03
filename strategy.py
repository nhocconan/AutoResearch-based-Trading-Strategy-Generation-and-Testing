#!/usr/bin/env python3
"""
Experiment #394: 1h Donchian Breakout + 4h Volume Spike + 1d Trend Filter

HYPOTHESIS: 1h Donchian(20) breakouts with 4h volume confirmation (>1.5x average) 
and 1d trend filter (price > EMA50 on daily) captures momentum moves while reducing 
false breakouts. Using 4h/1h MTF avoids 1h noise while allowing timely entries. 
Session filter (08-20 UTC) reduces off-hours whipsaws. Target: 60-150 total trades 
over 4 years (15-37/year) to minimize fee drag. Discrete position sizing (0.20) 
controls drawdown in bear markets like 2022.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_donchian_vol_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for volume spike (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate volume ratio (current vs 20-period average) on 4h
    if len(df_4h) >= 20:
        vol_4h = df_4h['volume'].values
        vol_ma_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_4h = np.zeros(len(vol_4h))
        vol_ratio_4h[20:] = vol_4h[20:] / vol_ma_20[20:]
        vol_ratio_4h[:20] = 1.0  # Neutral for warmup
        vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    else:
        vol_ratio_4h_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Calculate Donchian channels (20-period) ===
    if n >= 20:
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
        donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
        donchian_upper[:19] = np.nan
        donchian_lower[:19] = np.nan
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # Pre-compute before loop
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ratio_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian lower (trailing stop for longs)
                if close[i] <= donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian upper (trailing stop for shorts)
                if close[i] >= donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper with volume confirmation and uptrend
        long_condition = (
            close[i] > donchian_upper[i] and  # Breakout above upper channel
            vol_ratio_4h_aligned[i] > 1.5 and  # Volume spike confirmation
            close[i] > ema_50_1d_aligned[i]   # Price above daily EMA50 (uptrend)
        )
        
        # Short: Price breaks below Donchian lower with volume confirmation and downtrend
        short_condition = (
            close[i] < donchian_lower[i] and  # Breakdown below lower channel
            vol_ratio_4h_aligned[i] > 1.5 and  # Volume spike confirmation
            close[i] < ema_50_1d_aligned[i]   # Price below daily EMA50 (downtrend)
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #394: 1h Donchian Breakout + 4h Volume Spike + 1d Trend Filter

HYPOTHESIS: 1h Donchian(20) breakouts with 4h volume confirmation (>1.5x average) 
and 1d trend filter (price > EMA50 on daily) captures momentum moves while reducing 
false breakouts. Using 4h/1h MTF avoids 1h noise while allowing timely entries. 
Session filter (08-20 UTC) reduces off-hours whipsaws. Target: 60-150 total trades 
over 4 years (15-37/year) to minimize fee drag. Discrete position sizing (0.20) 
controls drawdown in bear markets like 2022.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_donchian_vol_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for volume spike (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate volume ratio (current vs 20-period average) on 4h
    if len(df_4h) >= 20:
        vol_4h = df_4h['volume'].values
        vol_ma_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_4h = np.zeros(len(vol_4h))
        vol_ratio_4h[20:] = vol_4h[20:] / vol_ma_20[20:]
        vol_ratio_4h[:20] = 1.0  # Neutral for warmup
        vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    else:
        vol_ratio_4h_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Calculate Donchian channels (20-period) ===
    if n >= 20:
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
        donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
        donchian_upper[:19] = np.nan
        donchian_lower[:19] = np.nan
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # Pre-compute before loop
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ratio_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian lower (trailing stop for longs)
                if close[i] <= donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian upper (trailing stop for shorts)
                if close[i] >= donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper with volume confirmation and uptrend
        long_condition = (
            close[i] > donchian_upper[i] and  # Breakout above upper channel
            vol_ratio_4h_aligned[i] > 1.5 and  # Volume spike confirmation
            close[i] > ema_50_1d_aligned[i]   # Price above daily EMA50 (uptrend)
        )
        
        # Short: Price breaks below Donchian lower with volume confirmation and downtrend
        short_condition = (
            close[i] < donchian_lower[i] and  # Breakdown below lower channel
            vol_ratio_4h_aligned[i] > 1.5 and  # Volume spike confirmation
            close[i] < ema_50_1d_aligned[i]   # Price below daily EMA50 (downtrend)
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals