#!/usr/bin/env python3
"""
Experiment #1554: 1h Donchian(20) Breakout + 4h/1d Trend + Volume Spike + Session Filter
HYPOTHESIS: 1h Donchian breakouts aligned with 4h and 1d trend direction, confirmed by volume spikes (>1.5x average) and restricted to active session (08-20 UTC), capture high-probability swing trades in both bull and bear markets. Using 4h/1d for signal direction and 1h only for entry timing reduces false breakouts. Position size fixed at 0.20 to limit drawdown. Target: 60-150 total trades over 4 years (15-37/year) by requiring multi-timeframe confluence and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1554_1h_donchian20_4h1d_trend_vol_sess_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours from open_time (already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # EMA(21) for 4h trend
    ema_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    trend_4h = np.where(close_4h > ema_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # EMA(50) for 1d trend
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 1h Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for 1d EMA(50) and 1h indicators
    
    for i in range(warmup, n):
        # --- Session filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require BOTH 4h and 1d trend alignment (same direction)
        if trend_4h_aligned[i] == trend_1d_aligned[i] and trend_4h_aligned[i] != 0:
            trend_direction = trend_4h_aligned[i]  # 1 for uptrend, -1 for downtrend
            
            # Volume confirmation: require volume spike (> 1.5x average)
            volume_spike = vol_ratio[i] > 1.5
            
            if trend_direction > 0 and volume_spike:  # Uptrend
                # Breakout: price breaks above upper band
                if price > donch_high[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                else:
                    signals[i] = 0.0
            elif trend_direction < 0 and volume_spike:  # Downtrend
                # Breakout: price breaks below lower band
                if price < donch_low[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #1554: 1h Donchian(20) Breakout + 4h/1d Trend + Volume Spike + Session Filter
HYPOTHESIS: 1h Donchian breakouts aligned with 4h and 1d trend direction, confirmed by volume spikes (>1.5x average) and restricted to active session (08-20 UTC), capture high-probability swing trades in both bull and bear markets. Using 4h/1d for signal direction and 1h only for entry timing reduces false breakouts. Position size fixed at 0.20 to limit drawdown. Target: 60-150 total trades over 4 years (15-37/year) by requiring multi-timeframe confluence and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1554_1h_donchian20_4h1d_trend_vol_sess_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours from open_time (already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # EMA(21) for 4h trend
    ema_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    trend_4h = np.where(close_4h > ema_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # EMA(50) for 1d trend
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 1h Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for 1d EMA(50) and 1h indicators
    
    for i in range(warmup, n):
        # --- Session filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require BOTH 4h and 1d trend alignment (same direction)
        if trend_4h_aligned[i] == trend_1d_aligned[i] and trend_4h_aligned[i] != 0:
            trend_direction = trend_4h_aligned[i]  # 1 for uptrend, -1 for downtrend
            
            # Volume confirmation: require volume spike (> 1.5x average)
            volume_spike = vol_ratio[i] > 1.5
            
            if trend_direction > 0 and volume_spike:  # Uptrend
                # Breakout: price breaks above upper band
                if price > donch_high[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                else:
                    signals[i] = 0.0
            elif trend_direction < 0 and volume_spike:  # Downtrend
                # Breakout: price breaks below lower band
                if price < donch_low[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals