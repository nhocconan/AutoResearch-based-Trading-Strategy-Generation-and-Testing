#!/usr/bin/env python3
"""
Experiment #2315: 6h ATR Breakout + Weekly Trend Filter + Volume Spike
HYPOTHESIS: Breakouts above/below ATR-based channels on 6h capture sustained moves when aligned with weekly trend.
- Primary: 6h ATR(14) channel breakout (close > upper band = long, close < lower band = short)
- HTF: 1w EMA(50) trend filter (only trade breakouts in direction of weekly trend)
- Entry: Long when close > upper band (mean + 2*ATR) with 1w uptrend + volume spike
         Short when close < lower band (mean - 2*ATR) with 1w downtrend + volume spike
- Exit: Opposite band touch or ATR(14) stop (2*ATR against position)
- Volume: Require > 1.5x 20-bar average spike to confirm participation (moderate to avoid overtrading)
- Target: 75-150 total trades over 4 years (19-37/year) - suitable for 6h timeframe
- Works in bull markets (breakouts with uptrend) and bear markets (breakouts with downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2315_6h_atr_breakout_1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for EMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50)
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 6h Indicators: ATR(14) channel, Volume MA(20) ===
    # ATR(14) calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 20-period SMA for channel middle line
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # ATR bands: middle ± 2*ATR
    upper_band = sma_20 + 2.0 * atr
    lower_band = sma_20 - 2.0 * atr
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
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
        if (np.isnan(trend_1w_aligned[i]) or np.isnan(upper_band[i]) or
            np.isnan(lower_band[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches lower band (mean reversion)
                elif price <= lower_band[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches upper band (mean reversion)
                elif price >= upper_band[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Weekly trend filter: only trade in direction of weekly trend
        trend_bias = trend_1w_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike and trend_bias != 0:
            # Long entry: price breaks above upper band with weekly uptrend
            if trend_bias > 0 and price > upper_band[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below lower band with weekly downtrend
            elif trend_bias < 0 and price < lower_band[i]:
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
Experiment #2315: 6h ATR Breakout + Weekly Trend Filter + Volume Spike
HYPOTHESIS: Breakouts above/below ATR-based channels on 6h capture sustained moves when aligned with weekly trend.
- Primary: 6h ATR(14) channel breakout (close > upper band = long, close < lower band = short)
- HTF: 1w EMA(50) trend filter (only trade breakouts in direction of weekly trend)
- Entry: Long when close > upper band (mean + 2*ATR) with 1w uptrend + volume spike
         Short when close < lower band (mean - 2*ATR) with 1w downtrend + volume spike
- Exit: Opposite band touch or ATR(14) stop (2*ATR against position)
- Volume: Require > 1.5x 20-bar average spike to confirm participation (moderate to avoid overtrading)
- Target: 75-150 total trades over 4 years (19-37/year) - suitable for 6h timeframe
- Works in bull markets (breakouts with uptrend) and bear markets (breakouts with downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2315_6h_atr_breakout_1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for EMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50)
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 6h Indicators: ATR(14) channel, Volume MA(20) ===
    # ATR(14) calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 20-period SMA for channel middle line
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # ATR bands: middle ± 2*ATR
    upper_band = sma_20 + 2.0 * atr
    lower_band = sma_20 - 2.0 * atr
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
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
        if (np.isnan(trend_1w_aligned[i]) or np.isnan(upper_band[i]) or
            np.isnan(lower_band[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches lower band (mean reversion)
                elif price <= lower_band[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches upper band (mean reversion)
                elif price >= upper_band[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Weekly trend filter: only trade in direction of weekly trend
        trend_bias = trend_1w_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike and trend_bias != 0:
            # Long entry: price breaks above upper band with weekly uptrend
            if trend_bias > 0 and price > upper_band[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below lower band with weekly downtrend
            elif trend_bias < 0 and price < lower_band[i]:
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