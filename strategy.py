#!/usr/bin/env python3
"""
Experiment #6411: 6h Donchian(20) breakout + 1d weekly pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts with volume confirmation (>2.0x avg) and 1d weekly pivot levels (R2/S2 for continuation, R1/S1 for mean reversion) capture institutional order flow. In ranging markets, price tends to reverse at R1/S1 (weekly pivot support/resistance). In trending markets, breaks of R2/S2 indicate strong momentum with continuation bias. Volume confirmation filters false breakouts. Discrete sizing (0.25) balances profit potential and drawdown control. Target: 75-200 trades over 4 years. Works in bull via R2 breakouts with volume, in bear via S2 breakdowns with volume, and ranges via R1/S1 reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6411_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    
    # === HTF: 1d data for weekly pivot levels (using prior week's OHLC) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 5:  # Need at least 5 days for prior week
        # Calculate weekly OHLC from daily data (prior completed week)
        # Week: Monday to Sunday, use last 7 days but shift by 1 to get prior week
        weekly_high = pd.Series(df_1d['high']).rolling(window=7, min_periods=7).max().shift(1).values
        weekly_low = pd.Series(df_1d['low']).rolling(window=7, min_periods=7).min().shift(1).values
        weekly_close = pd.Series(df_1d['close']).rolling(window=7, min_periods=7).last().shift(1).values
        
        # Weekly pivot formulas (similar to Camarilla but different multipliers)
        # Pivot = (High + Low + Close) / 3
        # R1 = (2 * Pivot) - Low
        # S1 = (2 * Pivot) - High
        # R2 = Pivot + (High - Low)
        # S2 = Pivot - (High - Low)
        # R3 = High + 2*(Pivot - Low)
        # S3 = Low - 2*(High - Pivot)
        
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_r1 = (2 * weekly_pivot) - weekly_low
        weekly_s1 = (2 * weekly_pivot) - weekly_high
        weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
        weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
        
        # Align to 6h timeframe (shifted by 1 week for lookback safety)
        r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
        s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
        r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
        s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    else:
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
        r2_aligned = np.full(n, np.nan)
        s2_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = max(20, 20, 14) + 1  # Donchian, volume avg, ATR lookback + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (22:00-23:59 UTC) ---
        hour = hours[i]
        if 22 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks below Donchian low (failed breakout)
                # 3. Price retraces to S1 (profit taking in range)
                # 4. Price breaks below S2 (failed continuation)
                if price <= stop_price or price <= donchian_low[i] or price <= s1_aligned[i] or price < s2_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks above Donchian high (failed breakout)
                # 3. Price retraces to R1 (profit taking in range)
                # 4. Price breaks above R2 (failed continuation)
                if price >= stop_price or price >= donchian_high[i] or price >= r1_aligned[i] or price > r2_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 2.0  # Volume filter
        
        # Entry logic based on weekly pivot levels:
        # Long: 
        #   - Breakout above R2 with volume (strong continuation)
        #   - OR bounce from S1 with volume (mean reversion in range)
        # Short:
        #   - Breakdown below S2 with volume (strong continuation)
        #   - OR rejection at R1 with volume (mean reversion in range)
        
        long_breakout = breakout_up and volume_confirmed and (price > r2_aligned[i])
        long_reversal = (price > s1_aligned[i]) and (close[i-1] <= s1_aligned[i-1]) and volume_confirmed  # Cross above S1
        
        short_breakout = breakout_down and volume_confirmed and (price < s2_aligned[i])
        short_reversal = (price < r1_aligned[i]) and (close[i-1] >= r1_aligned[i-1]) and volume_confirmed  # Cross below R1
        
        if long_breakout or long_reversal:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_breakout or short_reversal:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals