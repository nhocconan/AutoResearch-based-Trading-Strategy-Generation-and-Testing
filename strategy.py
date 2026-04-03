#!/usr/bin/env python3
"""
Experiment #155: 6h Williams %R + 1d Trend Filter + Volume Spike + ATR Stoploss

HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h timeframe. 
Entries occur when %R crosses above -50 from below (bullish momentum) or crosses below -50 from above (bearish momentum), 
filtered by 1d EMA200 trend alignment and confirmed by volume spikes (>2x average). 
This captures momentum continuations in both bull and bear markets while avoiding counter-trend trades. 
ATR-based stoploss (2.5x) manages risk. Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) 
to minimize fee drag while capturing high-probability momentum moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(200) on 1d close
    if len(df_1d) >= 200:
        close_1d = df_1d['close'].values
        ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    else:
        ema_200_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Williams %R(14) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.full(n, np.nan)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    denominator = highest_high - lowest_low
    # Avoid division by zero
    mask = denominator != 0
    williams_r[mask] = ((highest_high[mask] - close[mask]) / denominator[mask]) * -100
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 200  # Ensure enough data for HTF EMA200 and indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in direction of 1d EMA200 ---
        price_above_1d_ema = close[i] > ema_200_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_200_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Williams %R Signals ---
        # Bullish: %R crosses above -50 from below (momentum building)
        bullish_cross = (williams_r[i] > -50) and (williams_r[i-1] <= -50)
        # Bearish: %R crosses below -50 from above (momentum building)
        bearish_cross = (williams_r[i] < -50) and (williams_r[i-1] >= -50)
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Williams %R bullish cross + volume spike + price above 1d EMA200
        long_condition = bullish_cross and volume_spike and price_above_1d_ema
        
        # Short: Williams %R bearish cross + volume spike + price below 1d EMA200
        short_condition = bearish_cross and volume_spike and price_below_1d_ema
        
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
Experiment #155: 6h Williams %R + 1d Trend Filter + Volume Spike + ATR Stoploss

HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h timeframe. 
Entries occur when %R crosses above -50 from below (bullish momentum) or crosses below -50 from above (bearish momentum), 
filtered by 1d EMA200 trend alignment and confirmed by volume spikes (>2x average). 
This captures momentum continuations in both bull and bear markets while avoiding counter-trend trades. 
ATR-based stoploss (2.5x) manages risk. Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) 
to minimize fee drag while capturing high-probability momentum moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(200) on 1d close
    if len(df_1d) >= 200:
        close_1d = df_1d['close'].values
        ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    else:
        ema_200_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Williams %R(14) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.full(n, np.nan)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    denominator = highest_high - lowest_low
    # Avoid division by zero
    mask = denominator != 0
    williams_r[mask] = ((highest_high[mask] - close[mask]) / denominator[mask]) * -100
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 200  # Ensure enough data for HTF EMA200 and indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in direction of 1d EMA200 ---
        price_above_1d_ema = close[i] > ema_200_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_200_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Williams %R Signals ---
        # Bullish: %R crosses above -50 from below (momentum building)
        bullish_cross = (williams_r[i] > -50) and (williams_r[i-1] <= -50)
        # Bearish: %R crosses below -50 from above (momentum building)
        bearish_cross = (williams_r[i] < -50) and (williams_r[i-1] >= -50)
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Williams %R bullish cross + volume spike + price above 1d EMA200
        long_condition = bullish_cross and volume_spike and price_above_1d_ema
        
        # Short: Williams %R bearish cross + volume spike + price below 1d EMA200
        short_condition = bearish_cross and volume_spike and price_below_1d_ema
        
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