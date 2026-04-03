#!/usr/bin/env python3
"""
Experiment #274: 1h Camarilla Pivot Reversal + 4h/1d Trend Filter + Volume Spike

HYPOTHESIS: Camarilla pivot levels (L3, L4, H3, H4) act as strong support/resistance on 1h.
In ranging markets (ADX < 25), price reverses from these levels with volume confirmation.
In trending markets (ADX >= 25), we follow 4h/1d EMA trend direction for breakouts.
Session filter (08-20 UTC) reduces noise. Target: 60-150 trades over 4 years (15-37/year).
Works in bull (trend continuation) and bear (mean reversion + failed breaks) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_274_1h_camarilla_4h_1d_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h and 1d data for trend filters (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(21) on 4h and 1d for trend
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: Camarilla Pivot Levels (based on previous day) ===
    # Calculate daily OHLC from 1h data
    daily_open = np.full(n, np.nan)
    daily_high = np.full(n, np.nan)
    daily_low = np.full(n, np.nan)
    daily_close = np.full(n, np.nan)
    
    # Group by day using open_time (already datetime64)
    days = pd.DatetimeIndex(prices['open_time']).normalize()
    unique_days = np.unique(days)
    
    for day in unique_days:
        mask = (days == day)
        if not np.any(mask):
            continue
        idx = np.where(mask)[0]
        daily_open[idx] = open[mask][0] if 'open' in prices.columns else close[mask][0]
        daily_high[idx] = high[mask].max()
        daily_low[idx] = low[mask].min()
        daily_close[idx] = close[mask][-1]
    
    # Camarilla levels: based on previous day's range
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(1, n):
        if np.isnan(daily_high[i-1]) or np.isnan(daily_low[i-1]) or np.isnan(daily_close[i-1]):
            continue
        range_val = daily_high[i-1] - daily_low[i-1]
        camarilla_h3[i] = daily_close[i-1] + range_val * 1.1 / 4
        camarilla_l3[i] = daily_close[i-1] - range_val * 1.1 / 4
        camarilla_h4[i] = daily_close[i-1] + range_val * 1.1 / 2
        camarilla_l4[i] = daily_close[i-1] - range_val * 1.1 / 2
    
    # === 1h Indicators: ADX(14) for regime detection ===
    def calculate_dmi(high, low, close, period):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0) if high[i] - high[i-1] > low[i-1] - low[i] else 0
            minus_dm[i] = max(low[i-1] - low[i], 0) if low[i-1] - low[i] > high[i] - high[i-1] else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        plus_dm[0] = 0
        minus_dm[0] = 0
        tr[0] = high[0] - low[0]
        
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        return adx, plus_di, minus_di
    
    adx, plus_di, minus_di = calculate_dmi(high, low, close, 14)
    adx_mask = adx > 25  # Trending regime
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Session filter: 08-20 UTC ===
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # Skip if outside trading session
        if not session_mask[i]:
            signals[i] = 0.0
            continue
            
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Detection ---
        is_trending = adx_mask[i]
        is_ranging = ~adx_mask[i]
        
        # --- Price Action ---
        price = close[i]
        price_above_ema_4h = price > ema_4h_aligned[i]
        price_above_ema_1d = price > ema_1d_aligned[i]
        
        # --- Volume Confirmation ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss using 1.5*ATR) ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR for stoploss
            tr = max(high[i] - low[i], abs(price - close[i-1]), abs(low[i] - close[i-1]))
            atr = pd.Series([tr]).ewm(span=14, min_periods=14, adjust=False).mean().iloc[0] if i >= 14 else 0
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 1.5 * atr
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 2R
                if price >= entry_price + 3.0 * atr:
                    signals[i] = SIZE * 0.5  # Half position
                    continue
            else:  # Short position
                stop_level = entry_price + 1.5 * atr
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 2R
                if price <= entry_price - 3.0 * atr:
                    signals[i] = -SIZE * 0.5  # Half position
                    continue
            
            # Minimum holding period of 1 bar
            if bars_since_entry < 1:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Ranging market: Mean reversion from Camarilla levels
        if is_ranging:
            # Long: Price at L3/L4 with volume spike and bullish bias
            long_condition = (
                (price <= camarilla_l3[i] * 1.002 or price <= camarilla_l4[i] * 1.002) and
                volume_spike and
                price > close[i-1]  # Bullish price action
            )
            
            # Short: Price at H3/H4 with volume spike and bearish bias
            short_condition = (
                (price >= camarilla_h3[i] * 0.998 or price >= camarilla_h4[i] * 0.998) and
                volume_spike and
                price < close[i-1]  # Bearish price action
            )
        
        # Trending market: Follow 4h/1d EMA trend with pullback entries
        else:
            # Long: Pullback to EMA in uptrend
            long_condition = (
                price_above_ema_4h and price_above_ema_1d and
                price <= ema_4h_aligned[i] * 1.01 and  # Within 1% of EMA
                volume_spike and
                plus_di[i] > minus_di[i]  # Bullish momentum
            )
            
            # Short: Pullback to EMA in downtrend
            short_condition = (
                not price_above_ema_4h and not price_above_ema_1d and
                price >= ema_4h_aligned[i] * 0.99 and  # Within 1% of EMA
                volume_spike and
                minus_di[i] > plus_di[i]  # Bearish momentum
            )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = price
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = price
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals