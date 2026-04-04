#!/usr/bin/env python3
"""
Experiment #6056: 6h Elder Ray Index + 1w ADX regime filter + volume confirmation
HYPOTHESIS: Elder Ray (Bull/Bear Power) measures bull/bear strength relative to EMA13. 
In trending regimes (ADX>25 on 1w), trade in direction of Elder Ray with volume confirmation (>1.3x avg).
In ranging regimes (ADX<20), fade extreme Elder Ray readings. Uses discrete sizing (0.25) 
and ATR(14) trailing stop. Target: 75-150 trades over 4 years (19-38/year).
Works in bull markets (strong Bull Power in uptrend) and bear markets (strong Bear Power in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6056_6h_elder_ray_1w_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1w data for ADX regime filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 30:
        # Calculate ADX on weekly data
        wh = df_1w['high'].values
        wl = df_1w['low'].values
        wc = df_1w['close'].values
        
        # True Range
        w_tr1 = wh - wl
        w_tr2 = np.abs(wh - np.roll(wc, 1))
        w_tr3 = np.abs(wl - np.roll(wc, 1))
        w_tr = np.maximum(w_tr1, np.maximum(w_tr2, w_tr3))
        w_tr[0] = w_tr1[0]
        
        # Directional Movement
        w_dm_plus = np.where((wh - np.roll(wh, 1)) > (np.roll(wl, 1) - wl), 
                            np.maximum(wh - np.roll(wh, 1), 0), 0)
        w_dm_minus = np.where((np.roll(wl, 1) - wl) > (wh - np.roll(wh, 1)), 
                             np.maximum(np.roll(wl, 1) - wl, 0), 0)
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                result[period-1] = np.nansum(data[:period])
                for i in range(period, len(data)):
                    result[i] = result[i-1] - (result[i-1]/period) + data[i]
            return result
        
        period = 14
        w_atr = wilder_smooth(w_tr, period)
        w_di_plus = 100 * wilder_smooth(w_dm_plus, period) / np.where(w_atr > 0, w_atr, 1)
        w_di_minus = 100 * wilder_smooth(w_dm_minus, period) / np.where(w_atr > 0, w_atr, 1)
        w_dx = 100 * np.abs(w_di_plus - w_di_minus) / np.where((w_di_plus + w_di_minus) > 0, (w_di_plus + w_di_minus), 1)
        adx_1w = wilder_smooth(w_dx, period)
        adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    else:
        adx_1w_aligned = np.full(n, 20.0)  # Default to ranging
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # === 6h Indicators: Elder Ray Index (Bull Power & Bear Power) ===
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 6h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(30, 13, 20, 14, 1) + 1  # 1w ADX, EMA13, volume avg, ATR + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit: stoploss OR Elder Ray turns negative (loss of bullish strength)
                if price <= stop_price or bull_power[i] < 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit: stoploss OR Elder Ray turns positive (loss of bearish strength)
                if price >= stop_price or bear_power[i] > 0:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        adx_val = adx_1w_aligned[i]
        vol_confirmed = volume_ratio[i] > 1.3
        
        # Regime-based entry logic
        if adx_val > 25:  # Trending regime
            # Trend following: trade in direction of Elder Ray
            long_entry = bull_power[i] > 0 and vol_confirmed
            short_entry = bear_power[i] < 0 and vol_confirmed
        elif adx_val < 20:  # Ranging regime
            # Mean reversion: fade extreme Elder Ray readings
            long_entry = bear_power[i] < -0.5 * np.std(bear_power[max(0,i-100):i+1]) and vol_confirmed
            short_entry = bull_power[i] > 0.5 * np.std(bull_power[max(0,i-100):i+1]) and vol_confirmed
        else:  # Transition regime (20 <= ADX <= 25)
            # Require stronger confirmation
            long_entry = bull_power[i] > 0.2 * np.std(bull_power[max(0,i-100):i+1]) and vol_confirmed
            short_entry = bear_power[i] < -0.2 * np.std(bear_power[max(0,i-100):i+1]) and vol_confirmed
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>