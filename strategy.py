#!/usr/bin/env python3
"""
Experiment #5067: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Spike + ATR Stoploss
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot levels (from 1d HTF) capture strong momentum with lower frequency. Weekly pivot acts as regime filter: R3/S3 for mean reversion, R4/S4 for breakout confirmation. Volume > 2x average confirms institutional participation. ATR(14) trailing stop (2.5x) manages risk. Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance. Weekly pivot provides structural support/resistance that works in both bull (breakouts through R4) and bear (breakdowns through S4) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5067_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for weekly pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Weekly Pivot Points (using prior week's OHLC) ===
    if len(df_1d) >= 5:  # Need at least a week of data
        # Calculate weekly OHLC from daily data
        # We'll use rolling window of 5 days to approximate weekly OHLC
        # For true weekly pivot, we need prior week's H, L, C
        high_5d = pd.Series(high).rolling(window=5, min_periods=5).max().values
        low_5d = pd.Series(low).rolling(window=5, min_periods=5).min().values
        close_5d = pd.Series(close).rolling(window=5, min_periods=5).last().values
        
        # Weekly Pivot Point = (Prior Week H + L + C) / 3
        pp = (high_5d + low_5d + close_5d) / 3.0
        
        # Weekly Support/Resistance Levels
        # R1 = (2 * PP) - Prior Week L
        # S1 = (2 * PP) - Prior Week H
        # R2 = PP + (Prior Week H - Prior Week L)
        # S2 = PP - (Prior Week H - Prior Week L)
        # R3 = Prior Week H + 2*(PP - Prior Week L)
        # S3 = Prior Week L - 2*(Prior Week H - PP)
        # R4 = PP + 3*(Prior Week H - Prior Week L)
        # S4 = PP - 3*(Prior Week H - Prior Week L)
        rng = high_5d - low_5d
        r1 = (2 * pp) - low_5d
        s1 = (2 * pp) - high_5d
        r2 = pp + rng
        s2 = pp - rng
        r3 = high_5d + 2 * (pp - low_5d)
        s3 = low_5d - 2 * (high_5d - pp)
        r4 = pp + 3 * rng
        s4 = pp - 3 * rng
        
        # For breakout confirmation, we'll use R4/S4 levels
        # For mean reversion fade, we'll use R3/S3 levels
        # Align to 6h timeframe
        pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        pp_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (2x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>2.0x)
        vol_confirm = vol_ratio[i] > 2.0
        
        # Donchian breakout conditions with weekly pivot alignment
        # Long: Donchian breakout above R4 (strong breakout) OR above R3 with volume (mean reversion fail)
        # Short: Donchian breakdown below S4 (strong breakdown) OR below S3 with volume (mean reversion fail)
        breakout_long = ((price >= high_roll[i]) and 
                        ((price >= r4_aligned[i]) or  # Strong breakout through weekly R4
                         ((price >= r3_aligned[i]) and vol_confirm)) and  # Fade failure at R3 with volume
                        vol_confirm)
        
        breakout_short = ((price <= low_roll[i]) and 
                         ((price <= s4_aligned[i]) or  # Strong breakdown through weekly S4
                          ((price <= s3_aligned[i]) and vol_confirm)) and  # Fade failure at S3 with volume
                         vol_confirm)
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
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
#!/usr/bin/env python3
"""
Experiment #5067: 6h Williams %R Reversal + 1d ADX Trend Filter + Volume Confirmation
HYPOTHESIS: On 6h timeframe, Williams %R(14) extreme readings (<-80 for oversold, >-20 for overbought) 
combined with 1d ADX(14) > 25 to filter for trending markets captures high-probability reversals. 
Volume > 1.5x average confirms participation. Designed for 12-37 trades/year on 6h to minimize fee drag. 
Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend) markets via ADX filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5067_6h_williamsr_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: ADX(14) for trend strength ===
    if len(df_1d) >= 14:
        # Calculate True Range
        tr1 = df_1d['high'][1:] - df_1d['low'][1:]
        tr2 = np.abs(df_1d['high'][1:] - df_1d['close'][:-1])
        tr3 = np.abs(df_1d['low'][1:] - df_1d['close'][:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_1d = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Calculate +DM and -DM
        up_move = df_1d['high'][1:] - df_1d['high'][:-1]
        down_move = df_1d['low'][:-1] - df_1d['low'][1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed +DM, -DM, and TR
        tr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        plus_dm_14 = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
        minus_dm_14 = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Calculate +DI and -DI
        plus_di_14 = 100 * plus_dm_14 / tr_14
        minus_di_14 = 100 * minus_dm_14 / tr_14
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
        adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Align ADX to 6h timeframe
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Williams %R(14) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(14, 20, 14)  # Williams %R, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Williams %R extreme readings with ADX trend filter
        # Long: Williams %R < -80 (oversold) AND ADX > 25 (trending market) AND volume confirmation
        # Short: Williams %R > -20 (overbought) AND ADX > 25 (trending market) AND volume confirmation
        long_signal = (williams_r[i] < -80) and (adx_aligned[i] > 25) and vol_confirm
        short_signal = (williams_r[i] > -20) and (adx_aligned[i] > 25) and vol_confirm
        
        # Final entry conditions
        if long_signal:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_signal:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals