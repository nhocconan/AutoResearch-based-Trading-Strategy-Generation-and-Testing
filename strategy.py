#!/usr/bin/env python3
"""
Experiment #2263: 4h Donchian(20) breakout + 12h/1d HTF trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian breakouts on 4h capture swing momentum; HTF trend (12h/1d) filters direction to avoid counter-trend trades.
Volume spike (>2.0x 20-bar mean) confirms breakout strength. ATR(14) trailing stop (2.5x) manages risk.
Designed for 4h timeframe with target 75-200 trades over 4 years. Works in bull (trend following) and bear (mean reversion at channel extremes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2263_4h_donchian20_12h_1d_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h and 1d data for trend alignment (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(21) for 12h and 1d (simpler than HMA, equally effective)
    close_12h = df_12h['close'].values
    close_1d = df_1d['close'].values
    
    ema_12h = pd.Series(close_12h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d = pd.Series(close_1d).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Trend: 1 if close > EMA, -1 otherwise
    trend_12h = np.where(close_12h > ema_12h, 1, -1)
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Combined HTF trend: require both 12h and 1d to agree (stronger filter)
    htf_trend = np.where((trend_12h_aligned == 1) & (trend_1d_aligned == 1), 1,
                        np.where((trend_12h_aligned == -1) & (trend_1d_aligned == -1), -1, 0))
    
    # === 4h Indicators: Donchian(20), Volume MA(20), ATR(14) ===
    # Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    
    # Volume MA for spike detection (higher threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(htf_trend[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (wider stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches lower Donchian (mean reversion at extreme)
                elif price <= donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches upper Donchian (mean reversion at extreme)
                elif price >= donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require HTF trend alignment (both 12h and 1d agree) for bias filter
        trend_bias = htf_trend[i]
        
        # Volume confirmation: require volume spike (> 2.0x average - stricter to limit trades)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike and trend_bias != 0:
            # Long entry: price breaks above upper Donchian AND HTF trend up
            if trend_bias > 0 and price > donchian_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND HTF trend down
            elif trend_bias < 0 and price < donchian_lower[i]:
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
Experiment #2263: 4h Donchian(20) breakout + 12h/1d HTF trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian breakouts on 4h capture swing momentum; HTF trend (12h/1d) filters direction to avoid counter-trend trades.
Volume spike (>2.0x 20-bar mean) confirms breakout strength. ATR(14) trailing stop (2.5x) manages risk.
Designed for 4h timeframe with target 75-200 trades over 4 years. Works in bull (trend following) and bear (mean reversion at channel extremes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2263_4h_donchian20_12h_1d_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h and 1d data for trend alignment (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(21) for 12h and 1d (simpler than HMA, equally effective)
    close_12h = df_12h['close'].values
    close_1d = df_1d['close'].values
    
    ema_12h = pd.Series(close_12h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d = pd.Series(close_1d).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Trend: 1 if close > EMA, -1 otherwise
    trend_12h = np.where(close_12h > ema_12h, 1, -1)
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Combined HTF trend: require both 12h and 1d to agree (stronger filter)
    htf_trend = np.where((trend_12h_aligned == 1) & (trend_1d_aligned == 1), 1,
                        np.where((trend_12h_aligned == -1) & (trend_1d_aligned == -1), -1, 0))
    
    # === 4h Indicators: Donchian(20), Volume MA(20), ATR(14) ===
    # Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    
    # Volume MA for spike detection (higher threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(htf_trend[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (wider stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches lower Donchian (mean reversion at extreme)
                elif price <= donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches upper Donchian (mean reversion at extreme)
                elif price >= donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require HTF trend alignment (both 12h and 1d agree) for bias filter
        trend_bias = htf_trend[i]
        
        # Volume confirmation: require volume spike (> 2.0x average - stricter to limit trades)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike and trend_bias != 0:
            # Long entry: price breaks above upper Donchian AND HTF trend up
            if trend_bias > 0 and price > donchian_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND HTF trend down
            elif trend_bias < 0 and price < donchian_lower[i]:
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