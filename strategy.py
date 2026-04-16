#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1w EMA34 trend filter + 1d volume spike
# Uses Alligator jaws/teeth/lips for trend direction and exhaustion signals
# Long when lips > teeth > jaws (bullish alignment) AND price > lips AND 1w EMA34 uptrend AND 1d volume > 2.0x 20-median
# Short when lips < teeth < jaws (bearish alignment) AND price < lips AND 1w EMA34 downtrend AND 1d volume > 2.0x 20-median
# Exit when Alligator lines cross (jaws > lips for long exit, jaws < lips for short exit) or ATR stop (2.5)
# Position size 0.25 to balance capture and fee drag. Target: 80-180 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data once before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # === 1w Indicators ===
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get 1d data once before loop for Alligator and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d Indicators ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Williams Alligator: SMAs of median price (HL/2) with specific periods
    # Jaws: SMA(13, 8) - 13-period SMA, 8 bars ahead
    # Teeth: SMA(8, 5) - 8-period SMA, 5 bars ahead  
    # Lips: SMA(5, 3) - 5-period SMA, 3 bars ahead
    median_price_1d = (high_1d + low_1d) / 2.0
    
    # Calculate SMAs
    sma5 = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    sma8 = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    sma13 = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    
    # Apply forward shift (Alligator specific)
    jaws = np.concatenate([np.full(8, np.nan), sma13[:-8]])  # 8 bars ahead
    teeth = np.concatenate([np.full(5, np.nan), sma8[:-5]])   # 5 bars ahead
    lips = np.concatenate([np.full(3, np.nan), sma5[:-3]])    # 3 bars ahead
    
    # Align Alligator lines to 6h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 1d volume median (20-period) for spike detection
    vol_median_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    vol_median_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20_1d)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(40, 60, 13, 20)  # 1w EMA34, 1d Alligator, volume median
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0  # force flat outside session
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_median_20_1d_aligned[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current 1d volume (aligned)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        if np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 2.0x 20-period 1d volume median
        vol_threshold = vol_median_20_1d_aligned[i] * 2.0
        vol_confirm = vol_1d_aligned[i] > vol_threshold
        
        # Trend filter: 1w EMA34 direction
        trend_up = close[i] > ema34_1w_aligned[i]
        trend_down = close[i] < ema34_1w_aligned[i]
        
        # Alligator alignment conditions
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaws_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaws_aligned[i]
        
        # Price vs lips for entry confirmation
        price = close[i]
        lips_level = lips_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit on Alligator bearish cross (jaws > lips) or ATR stoploss
            if jaws_aligned[i] > lips_aligned[i] or price <= entry_price - 2.5 * atr_14[i]:
                exit_signal = True
        elif position == -1:  # short position
            # Exit on Alligator bullish cross (jaws < lips) or ATR stoploss
            if jaws_aligned[i] < lips_aligned[i] or price >= entry_price + 2.5 * atr_14[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Bullish Alligator alignment AND price above lips AND uptrend AND volume confirmation
            if bullish_alignment and price > lips_level and trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Bearish Alligator alignment AND price below lips AND downtrend AND volume confirmation
            elif bearish_alignment and price < lips_level and trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = 0.0  # maintain position
    
    return signals

name = "6h_WilliamsAlligator_1wEMA34_1dVolSpike_v1"
timeframe = "6h"
leverage = 1.0