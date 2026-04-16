#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w trend filter (EMA34) and volume confirmation.
# Long when price breaks above Camarilla R3 AND close > 1w EMA34 AND volume > 2.0x 20-period average.
# Short when price breaks below Camarilla S3 AND close < 1w EMA34 AND volume > 2.0x 20-period average.
# Exit when price returns to Camarilla H4/L4 levels or ATR(14) < ATR(50) (contracting volatility).
# Uses discrete position size 0.25. Camarilla provides intraday structure, 1w EMA34 filters major trend,
# volume confirms breakout strength, ATR regime filter avoids false breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: EMA34 for trend filter ===
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels calculation (based on previous day)
    # H4 = Close + 1.1 * (High - Low) / 2
    # L4 = Close - 1.1 * (High - Low) / 2
    # R3 = Close + 1.1 * (High - Low) / 2
    # S3 = Close - 1.1 * (High - Low) / 2
    # Note: For intraday, H4=R3 and L4=S3 in standard Camarilla
    camarilla_range = high_1d - low_1d
    camarilla_h4 = close_1d + 1.1 * camarilla_range / 2.0
    camarilla_l4 = close_1d - 1.1 * camarilla_range / 2.0
    camarilla_r3 = camarilla_h4  # Same as H4
    camarilla_s3 = camarilla_l4  # Same as L4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume moving average (20-period) on 12h - using primary timeframe data directly
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # True Range for ATR calculation (using primary timeframe)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50) for regime filter
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        ema_34_val = ema_34_aligned[i]
        h4_val = camarilla_h4_aligned[i]
        l4_val = camarilla_l4_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        vol_ma_val = vol_ma_20[i]
        atr_14_val = atr_14[i]
        atr_50_val = atr_50[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to H4/L4 levels or ATR contracts
            if price <= h4_val or price >= l4_val or atr_14_val < atr_50_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to H4/L4 levels or ATR contracts
            if price >= h4_val or price <= l4_val or atr_14_val < atr_50_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 2.0x 20-period average
            vol_filter = vol > 2.0 * vol_ma_val
            
            # Trend filter: price relative to 1w EMA34
            trend_filter_long = price > ema_34_val
            trend_filter_short = price < ema_34_val
            
            # LONG: price breaks above Camarilla R3 with volume and trend confirmation
            if price > r3_val and vol_filter and trend_filter_long:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: price breaks below Camarilla S3 with volume and trend confirmation
            elif price < s3_val and vol_filter and trend_filter_short:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_CamarillaR3S3_1wEMA34_Volume_ATRRegime_V1"
timeframe = "12h"
leverage = 1.0