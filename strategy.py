#!/usr/bin/env python3
"""
Experiment #654: 1h RSI(2) mean reversion + 4h/1d trend filter + volume spike
HYPOTHESIS: In 1h timeframe, RSI(2) extremes (<10 for long, >90 for short) aligned with 4h/1d trend (price > EMA50) and volume spikes capture high-probability mean reversion moves. Using higher timeframes for direction reduces false signals and overtrading. Target: 15-37 trades/year per symbol with Sharpe > 0 on test.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_654_1h_rsi2_4h1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h and 1d data for trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    close_4h = df_4h['close'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h EMA(50) and 1d EMA(50) for trend filter
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align HTF EMAs to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: RSI(2) for mean reversion signals ===
    def calculate_rsi(arr, period):
        delta = np.diff(arr, prepend=arr[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_2 = calculate_rsi(close, 2)
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Session filter: 08-20 UTC (reduce noise trades) ===
    # open_time is already datetime64[ms], access hour via index
    hours = prices.index.hour  # Pre-compute before loop
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # sufficient for RSI(2) and EMAs
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(rsi_2[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        hour = hours[i]
        
        # --- Session Filter: Only trade 08-20 UTC ---
        in_session = (8 <= hour <= 20)
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Trend Filter: Price above both 4h and 1d EMA50 for uptrend ---
        uptrend = price > ema_4h_aligned[i] and price > ema_1d_aligned[i]
        
        # --- Trend Filter: Price below both 4h and 1d EMA50 for downtrend ---
        downtrend = price < ema_4h_aligned[i] and price < ema_1d_aligned[i]
        
        # --- Mean Reversion Entry Logic ---
        if in_session and volume_spike:
            # Long: RSI(2) < 10 (extreme oversold) + uptrend on higher timeframes
            if rsi_2[i] < 10 and uptrend:
                if not in_position or position_side == -1:
                    # Reverse or enter long
                    in_position = True
                    position_side = 1
                    entry_price = price
                    signals[i] = SIZE
                else:
                    signals[i] = SIZE  # Already long, maintain
            # Short: RSI(2) > 90 (extreme overbought) + downtrend on higher timeframes
            elif rsi_2[i] > 90 and downtrend:
                if not in_position or position_side == 1:
                    # Reverse or enter short
                    in_position = True
                    position_side = -1
                    entry_price = price
                    signals[i] = -SIZE
                else:
                    signals[i] = -SIZE  # Already short, maintain
            else:
                # No mean reversion signal, flatten if not already flat
                if in_position:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.0
        else:
            # Outside session or no volume spike, flatten position
            if in_position:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #654: 1h RSI(2) mean reversion + 4h/1d trend filter + volume spike
HYPOTHESIS: In 1h timeframe, RSI(2) extremes (<10 for long, >90 for short) aligned with 4h/1d trend (price > EMA50) and volume spikes capture high-probability mean reversion moves. Using higher timeframes for direction reduces false signals and overtrading. Target: 15-37 trades/year per symbol with Sharpe > 0 on test.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_654_1h_rsi2_4h1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h and 1d data for trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    close_4h = df_4h['close'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h EMA(50) and 1d EMA(50) for trend filter
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align HTF EMAs to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: RSI(2) for mean reversion signals ===
    def calculate_rsi(arr, period):
        delta = np.diff(arr, prepend=arr[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_2 = calculate_rsi(close, 2)
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Session filter: 08-20 UTC (reduce noise trades) ===
    # open_time is already datetime64[ms], access hour via index
    hours = prices.index.hour  # Pre-compute before loop
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # sufficient for RSI(2) and EMAs
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(rsi_2[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        hour = hours[i]
        
        # --- Session Filter: Only trade 08-20 UTC ---
        in_session = (8 <= hour <= 20)
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Trend Filter: Price above both 4h and 1d EMA50 for uptrend ---
        uptrend = price > ema_4h_aligned[i] and price > ema_1d_aligned[i]
        
        # --- Trend Filter: Price below both 4h and 1d EMA50 for downtrend ---
        downtrend = price < ema_4h_aligned[i] and price < ema_1d_aligned[i]
        
        # --- Mean Reversion Entry Logic ---
        if in_session and volume_spike:
            # Long: RSI(2) < 10 (extreme oversold) + uptrend on higher timeframes
            if rsi_2[i] < 10 and uptrend:
                if not in_position or position_side == -1:
                    # Reverse or enter long
                    in_position = True
                    position_side = 1
                    entry_price = price
                    signals[i] = SIZE
                else:
                    signals[i] = SIZE  # Already long, maintain
            # Short: RSI(2) > 90 (extreme overbought) + downtrend on higher timeframes
            elif rsi_2[i] > 90 and downtrend:
                if not in_position or position_side == 1:
                    # Reverse or enter short
                    in_position = True
                    position_side = -1
                    entry_price = price
                    signals[i] = -SIZE
                else:
                    signals[i] = -SIZE  # Already short, maintain
            else:
                # No mean reversion signal, flatten if not already flat
                if in_position:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.0
        else:
            # Outside session or no volume spike, flatten position
            if in_position:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals