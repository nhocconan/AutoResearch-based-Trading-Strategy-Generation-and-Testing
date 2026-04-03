#!/usr/bin/env python3
"""
Experiment #011: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: Donchian channel breakouts on 6h timeframe capture medium-term momentum. 
Filtered by 1d weekly pivot direction (price above/below weekly pivot from prior week) 
and confirmed by 6h volume spike (>1.5x 20-period average). This avoids false breakouts 
in ranging markets while catching strong trends. Weekly pivot acts as regime filter: 
only long when price > weekly PP, short when price < weekly PP. 
Target: 12-37 trades/year on 6h (50-150 total over 4 years) to minimize fee drag. 
Uses discrete position sizing (0.25) and ATR-based stoploss (2.0x ATR) for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot and volume confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # === Calculate weekly pivot from prior week (using daily OHLC) ===
    # Group daily data into weeks (starting Monday) and get weekly high/low/close
    if len(df_1d) >= 5:  # Need at least a week of data
        # Create a DataFrame for resampling (temporarily)
        df_1d_temp = pd.DataFrame({
            'open': df_1d['open'].values,
            'high': df_1d['high'].values,
            'low': df_1d['low'].values,
            'close': df_1d['close'].values
        }, index=pd.to_datetime(df_1d['open_time']))
        
        # Resample to weekly (W-MON: weekly starting Monday)
        df_weekly = df_1d_temp.resample('W-MON').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
        
        if len(df_weekly) >= 2:
            # Calculate weekly pivot points from prior week
            weekly_high = df_weekly['high'].values[:-1]  # Prior week
            weekly_low = df_weekly['low'].values[:-1]
            weekly_close = df_weekly['close'].values[:-1]
            
            weekly_p = (weekly_high + weekly_low + weekly_close) / 3.0
            weekly_range = weekly_high - weekly_low
            
            # For Camarilla-like weekly levels, we use standard pivot levels
            # Weekly R1 = PP + (H-L), Weekly S1 = PP - (H-L)
            weekly_r1 = weekly_p + weekly_range
            weekly_s1 = weekly_p - weekly_range
            
            # Align weekly data to 1d timeframe (each weekly value applies to 5 trading days)
            # Forward fill weekly values to daily
            weekly_p_daily = np.repeat(weekly_p, 5)[:len(df_1d)]
            weekly_r1_daily = np.repeat(weekly_r1, 5)[:len(df_1d)]
            weekly_s1_daily = np.repeat(weekly_s1, 5)[:len(df_1d)]
            
            # Handle remainder days
            rem = len(df_1d) % 5
            if rem > 0 and len(weekly_p) > len(weekly_p_daily)//5:
                weekly_p_daily[-rem:] = weekly_p[-1]
                weekly_r1_daily[-rem:] = weekly_r1[-1]
                weekly_s1_daily[-rem:] = weekly_s1[-1]
            
            # Align to 6h timeframe
            weekly_p_aligned = align_htf_to_ltf(prices, df_1d, weekly_p_daily)
            weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1_daily)
            weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1_daily)
        else:
            weekly_p_aligned = np.full(n, np.nan)
            weekly_r1_aligned = np.full(n, np.nan)
            weekly_s1_aligned = np.full(n, np.nan)
    else:
        weekly_p_aligned = np.full(n, np.nan)
        weekly_r1_aligned = np.full(n, np.nan)
        weekly_s1_aligned = np.full(n, np.nan)
    
    # === Volume confirmation on 6h: volume > 1.5x 20-period average ===
    if len(volume) >= 20:
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_ratio = np.zeros(n)
        vol_ratio[20:] = volume[20:] / vol_ma[20:]
        vol_ratio[:20] = 1.0  # Neutral for warmup
    else:
        vol_ratio = np.full(n, 1.0)
    
    # === Donchian Channel (20) on 6h ===
    if len(high) >= 20 and len(low) >= 20:
        # Donchian Upper = 20-period high
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Donchian Lower = 20-period low
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_p_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Regime Filter: Weekly Pivot Direction ---
        price_above_weekly_p = close[i] > weekly_p_aligned[i]
        price_below_weekly_p = close[i] < weekly_p_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian Upper with volume confirmation and bullish regime
        long_condition = (
            close[i] > donchian_high[i] and 
            volume_spike and 
            price_above_weekly_p
        )
        
        # Short: Price breaks below Donchian Lower with volume confirmation and bearish regime
        short_condition = (
            close[i] < donchian_low[i] and 
            volume_spike and 
            price_below_weekly_p
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
Experiment #011: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: Donchian channel breakouts on 6h timeframe capture medium-term momentum. 
Filtered by 1d weekly pivot direction (price above/below weekly pivot from prior week) 
and confirmed by 6h volume spike (>1.5x 20-period average). This avoids false breakouts 
in ranging markets while catching strong trends. Weekly pivot acts as regime filter: 
only long when price > weekly PP, short when price < weekly PP. 
Target: 12-37 trades/year on 6h (50-150 total over 4 years) to minimize fee drag. 
Uses discrete position sizing (0.25) and ATR-based stoploss (2.0x ATR) for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot and volume confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # === Calculate weekly pivot from prior week (using daily OHLC) ===
    # Group daily data into weeks (starting Monday) and get weekly high/low/close
    if len(df_1d) >= 5:  # Need at least a week of data
        # Create a DataFrame for resampling (temporarily)
        df_1d_temp = pd.DataFrame({
            'open': df_1d['open'].values,
            'high': df_1d['high'].values,
            'low': df_1d['low'].values,
            'close': df_1d['close'].values
        }, index=pd.to_datetime(df_1d['open_time']))
        
        # Resample to weekly (W-MON: weekly starting Monday)
        df_weekly = df_1d_temp.resample('W-MON').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
        
        if len(df_weekly) >= 2:
            # Calculate weekly pivot points from prior week
            weekly_high = df_weekly['high'].values[:-1]  # Prior week
            weekly_low = df_weekly['low'].values[:-1]
            weekly_close = df_weekly['close'].values[:-1]
            
            weekly_p = (weekly_high + weekly_low + weekly_close) / 3.0
            weekly_range = weekly_high - weekly_low
            
            # For Camarilla-like weekly levels, we use standard pivot levels
            # Weekly R1 = PP + (H-L), Weekly S1 = PP - (H-L)
            weekly_r1 = weekly_p + weekly_range
            weekly_s1 = weekly_p - weekly_range
            
            # Align weekly data to 1d timeframe (each weekly value applies to 5 trading days)
            # Forward fill weekly values to daily
            weekly_p_daily = np.repeat(weekly_p, 5)[:len(df_1d)]
            weekly_r1_daily = np.repeat(weekly_r1, 5)[:len(df_1d)]
            weekly_s1_daily = np.repeat(weekly_s1, 5)[:len(df_1d)]
            
            # Handle remainder days
            rem = len(df_1d) % 5
            if rem > 0 and len(weekly_p) > len(weekly_p_daily)//5:
                weekly_p_daily[-rem:] = weekly_p[-1]
                weekly_r1_daily[-rem:] = weekly_r1[-1]
                weekly_s1_daily[-rem:] = weekly_s1[-1]
            
            # Align to 6h timeframe
            weekly_p_aligned = align_htf_to_ltf(prices, df_1d, weekly_p_daily)
            weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1_daily)
            weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1_daily)
        else:
            weekly_p_aligned = np.full(n, np.nan)
            weekly_r1_aligned = np.full(n, np.nan)
            weekly_s1_aligned = np.full(n, np.nan)
    else:
        weekly_p_aligned = np.full(n, np.nan)
        weekly_r1_aligned = np.full(n, np.nan)
        weekly_s1_aligned = np.full(n, np.nan)
    
    # === Volume confirmation on 6h: volume > 1.5x 20-period average ===
    if len(volume) >= 20:
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_ratio = np.zeros(n)
        vol_ratio[20:] = volume[20:] / vol_ma[20:]
        vol_ratio[:20] = 1.0  # Neutral for warmup
    else:
        vol_ratio = np.full(n, 1.0)
    
    # === Donchian Channel (20) on 6h ===
    if len(high) >= 20 and len(low) >= 20:
        # Donchian Upper = 20-period high
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Donchian Lower = 20-period low
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_p_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Regime Filter: Weekly Pivot Direction ---
        price_above_weekly_p = close[i] > weekly_p_aligned[i]
        price_below_weekly_p = close[i] < weekly_p_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian Upper with volume confirmation and bullish regime
        long_condition = (
            close[i] > donchian_high[i] and 
            volume_spike and 
            price_above_weekly_p
        )
        
        # Short: Price breaks below Donchian Lower with volume confirmation and bearish regime
        short_condition = (
            close[i] < donchian_low[i] and 
            volume_spike and 
            price_below_weekly_p
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