#!/usr/bin/env python3
"""
Experiment #054: 1h Camarilla Pivot Breakout + 4h Trend Filter + Volume Spike + Session Filter

HYPOTHESIS: Camarilla pivot breakouts on 1h timeframe, filtered by 4h trend alignment (price > EMA20) 
and confirmed by volume spikes (>2.0x average), capture intraday momentum with reduced false breakouts. 
4h trend filter ensures alignment with the intermediate trend, avoiding counter-trend trades in both bull and bear markets. 
Session filter (08-20 UTC) focuses on high-liquidity periods, reducing noise. 
Volume confirmation filters out low-conviction breakouts. Fixed position size (0.20) manages risk.
Target: 60-150 total trades over 4 years = 15-37/year for 1h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_054_1h_camarilla_4h_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values  # already datetime64[ms]
    n = len(close)
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate EMA(20) on 4h close
    if len(df_4h) >= 20:
        close_4h = df_4h['close'].values
        ema_20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
        ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    else:
        ema_20_4h_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Camarilla Pivot Levels (based on previous day) ===
    # Camarilla levels: H4, L4, H3, L3, H2, L2, H1, L1
    # We'll use H3 and L3 for breakout (most significant levels)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    
    # Calculate daily pivot from previous day's OHLC
    # We need to group by day to get previous day's high, low, close
    df = prices.copy()
    df['date'] = pd.DatetimeIndex(open_time).date
    daily = df.groupby('date').agg({'high': 'max', 'low': 'min', 'close': 'last'})
    
    # Shift to get previous day's values
    daily['prev_high'] = daily['high'].shift(1)
    daily['prev_low'] = daily['low'].shift(1)
    daily['prev_close'] = daily['close'].shift(1)
    
    # Calculate Camarilla levels for each day
    # H3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    # L3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    daily['cam_h3'] = daily['prev_close'] + 1.1 * (daily['prev_high'] - daily['prev_low']) / 4
    daily['cam_l3'] = daily['prev_close'] - 1.1 * (daily['prev_high'] - daily['prev_low']) / 4
    
    # Map back to original dataframe
    camarilla_map = daily[['cam_h3', 'cam_l3']].to_dict('index')
    camarilla_h3_values = []
    camarilla_l3_values = []
    
    for dt in pd.DatetimeIndex(open_time):
        date_key = dt.date()
        if date_key in camarilla_map:
            camarilla_h3_values.append(camarilla_map[date_key]['cam_h3'])
            camarilla_l3_values.append(camarilla_map[date_key]['cam_l3'])
        else:
            camarilla_h3_values.append(np.nan)
            camarilla_l3_values.append(np.nan)
    
    camarilla_h3 = np.array(camarilla_h3_values)
    camarilla_l3 = np.array(camarilla_l3_values)
    
    # === 1h Indicators: ATR(14) for stoploss reference (not used in entry, but for context) ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF EMA20 and ATR
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade between 08-20 UTC ---
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade when price above 4h EMA20 (long) or below (short) ---
        price_above_4h_ema = close[i] > ema_20_4h_aligned[i]
        price_below_4h_ema = close[i] < ema_20_4h_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Camarilla Breakout Conditions ---
        breakout_up = close[i] > camarilla_h3[i]
        breakout_down = close[i] < camarilla_l3[i]
        
        # --- Exit Logic: Fixed stoploss and take profit based on ATR ---
        if in_position:
            # Fixed stoploss: 2.0 * ATR
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 3.0 * ATR
                if close[i] >= entry_price + 3.0 * atr_14[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 3.0 * ATR
                if close[i] <= entry_price - 3.0 * atr_14[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Camarilla breakout up + volume spike + price above 4h EMA20
        long_condition = breakout_up and volume_spike and price_above_4h_ema
        
        # Short: Camarilla breakout down + volume spike + price below 4h EMA20
        short_condition = breakout_down and volume_spike and price_below_4h_ema
        
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