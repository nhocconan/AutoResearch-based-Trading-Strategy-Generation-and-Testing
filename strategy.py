#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter.
# Long when price breaks above Camarilla R4 AND 1d volume > 1.5x 20-period average AND 1w close > 1w EMA50 (bullish weekly trend).
# Short when price breaks below Camarilla S4 AND 1d volume > 1.5x 20-period average AND 1w close < 1w EMA50 (bearish weekly trend).
# Exit when price retests the Camarilla pivot point (PP) or ATR-based stoploss (1.5*ATR from entry).
# Uses discrete position size 0.25. Designed to capture institutional breakouts aligned with weekly trend.
# Target: 80-160 total trades over 4 years (20-40/year) to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Camarilla Pivot Levels (based on prior 6h bar) ===
    # Note: Camarilla uses prior period's OHLC
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Prior 6h bar's OHLC (shifted by 1 to avoid look-ahead)
    prior_high = np.roll(high_6h, 1)
    prior_low = np.roll(low_6h, 1)
    prior_close = np.roll(close_6h, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Camarilla calculations
    range_6h = prior_high - prior_low
    camarilla_pp = prior_close
    camarilla_r4 = prior_close + range_6h * 1.5
    camarilla_s4 = prior_close - range_6h * 1.5
    
    # Align to 6h timeframe (already aligned since we're using 6h data)
    camarilla_pp_aligned = camarilla_pp
    camarilla_r4_aligned = camarilla_r4
    camarilla_s4_aligned = camarilla_s4
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 1w Indicators: Trend Filter (close > EMA50 for long, close < EMA50 for short) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    weekly_bullish = close_1w > ema_50_1w_aligned  # For long bias
    weekly_bearish = close_1w < ema_50_1w_aligned  # For short bias
    
    # === 6h Indicators: ATR for stoploss ===
    tr1_6h = pd.Series(high_6h).diff()
    tr2_6h = pd.Series(low_6h).diff().abs()
    tr3_6h = pd.Series(close_6h).shift(1).diff().abs()
    tr_6h = pd.concat([tr1_6h, tr2_6h, tr3_6h], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h_raw)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA50)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or np.isnan(camarilla_pp_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_6h_aligned[i]) or
            np.isnan(weekly_bullish[i]) or np.isnan(weekly_bearish[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_6h_aligned[i]
        is_weekly_bullish = weekly_bullish[i]
        is_weekly_bearish = weekly_bearish[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price retests pivot point
            if price <= camarilla_pp_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price retests pivot point
            if price >= camarilla_pp_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR above entry
            elif price > entry_price + 1.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R4 AND volume spike AND weekly bullish
            if price > camarilla_r4_aligned[i] and vol_spike and is_weekly_bullish:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S4 AND volume spike AND weekly bearish
            elif price < camarilla_s4_aligned[i] and vol_spike and is_weekly_bearish:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Camarilla_R4_S4_1dVolumeSpike_1wEMA50_V1"
timeframe = "6h"
leverage = 1.0