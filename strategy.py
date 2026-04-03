#!/usr/bin/env python3
"""
Experiment #234: 1h RSI(2) extreme + 4h trend filter + volume spike
HYPOTHESIS: Intraday mean reversion at RSI(2) extremes (<10/>90) aligned with 4h EMA(50) trend direction captures high-probability reversals. Volume confirmation (>1.5x average) filters weak signals. Discrete sizing (0.20) minimizes fee drag. Target: 60-150 total trades over 4 years (15-37/year). Works in bull markets via buying weakness in uptrends and in bear markets via selling strength in downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_234_1h_rsi2_4h_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for EMA trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === HTF: 1d data for session alignment (optional regime) ===
    df_1d = get_htf_data(prices, '1d')
    # Not used directly but ensures HTF loading pattern is correct
    
    # === 1h Indicators: RSI(2) for mean reversion ===
    def calculate_rsi(series, period):
        delta = np.diff(series, prepend=series[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0.0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_2 = calculate_rsi(close, 2)
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Session filter: 08-20 UTC (precompute for performance) ===
    # open_time is already datetime64[ms], use DatetimeIndex
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if outside active session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # --- Data Validity Check ---
        if (np.isnan(rsi_2[i]) or np.isnan(ema_4h_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- RSI Extreme Conditions ---
        rsi_oversold = rsi_2[i] < 10
        rsi_overbought = rsi_2[i] > 90
        
        # --- 4h Trend Condition ---
        trend_up = close[i] > ema_4h_aligned[i]
        trend_down = close[i] < ema_4h_aligned[i]
        
        # --- Exit Logic: Mean reversion complete or adverse move ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Exit when RSI returns to neutral (50) or adverse move
                if rsi_2[i] >= 50 or (bars_since_entry >= 10 and close[i] < entry_price):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Exit when RSI returns to neutral (50) or adverse move
                if rsi_2[i] <= 50 or (bars_since_entry >= 10 and close[i] > entry_price):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position during mean reversion
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require volume spike + RSI extreme + 4h trend alignment
        if volume_spike:
            # Long: RSI oversold AND 4h trend up (buy weakness in uptrend)
            if rsi_oversold and trend_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: RSI overbought AND 4h trend down (sell strength in downtrend)
            elif rsi_overbought and trend_down:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #234: 1h RSI(2) extreme + 4h trend filter + volume spike
HYPOTHESIS: Intraday mean reversion at RSI(2) extremes (<10/>90) aligned with 4h EMA(50) trend direction captures high-probability reversals. Volume confirmation (>1.5x average) filters weak signals. Discrete sizing (0.20) minimizes fee drag. Target: 60-150 total trades over 4 years (15-37/year). Works in bull markets via buying weakness in uptrends and in bear markets via selling strength in downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_234_1h_rsi2_4h_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for EMA trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === HTF: 1d data for session alignment (optional regime) ===
    df_1d = get_htf_data(prices, '1d')
    # Not used directly but ensures HTF loading pattern is correct
    
    # === 1h Indicators: RSI(2) for mean reversion ===
    def calculate_rsi(series, period):
        delta = np.diff(series, prepend=series[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0.0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_2 = calculate_rsi(close, 2)
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Session filter: 08-20 UTC (precompute for performance) ===
    # open_time is already datetime64[ms], use DatetimeIndex
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if outside active session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # --- Data Validity Check ---
        if (np.isnan(rsi_2[i]) or np.isnan(ema_4h_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- RSI Extreme Conditions ---
        rsi_oversold = rsi_2[i] < 10
        rsi_overbought = rsi_2[i] > 90
        
        # --- 4h Trend Condition ---
        trend_up = close[i] > ema_4h_aligned[i]
        trend_down = close[i] < ema_4h_aligned[i]
        
        # --- Exit Logic: Mean reversion complete or adverse move ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Exit when RSI returns to neutral (50) or adverse move
                if rsi_2[i] >= 50 or (bars_since_entry >= 10 and close[i] < entry_price):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Exit when RSI returns to neutral (50) or adverse move
                if rsi_2[i] <= 50 or (bars_since_entry >= 10 and close[i] > entry_price):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position during mean reversion
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require volume spike + RSI extreme + 4h trend alignment
        if volume_spike:
            # Long: RSI oversold AND 4h trend up (buy weakness in uptrend)
            if rsi_oversold and trend_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: RSI overbought AND 4h trend down (sell strength in downtrend)
            elif rsi_overbought and trend_down:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals