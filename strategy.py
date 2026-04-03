#!/usr/bin/env python3
"""
Experiment #039: 6h Camarilla Pivot Fade/Breakout with 12h Trend Filter

HYPOTHESIS: On 6h timeframe, price reactions to 12h Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout) 
combined with 12h trend filter (price above/below 20-period EMA) capture mean reversion in ranges 
and continuation in trends. Volume confirmation (1.5x average) filters false signals. 
Designed for 12-37 trades/year on BTC/ETH/SOL to avoid fee drag in ranging/ bear markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_pivot_12h_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(values, period):
    """Calculate EMA with proper min_periods."""
    return pd.Series(values).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivots and EMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    h_close = df_12h['close'].values
    h_high = df_12h['high'].values
    h_low = df_12h['low'].values
    
    # Calculate typical price for pivot (using typical price = (H+L+C)/3)
    typical_price = (h_high + h_low + h_close) / 3.0
    # Calculate range
    hl_range = h_high - h_low
    
    # Camarilla levels for typical price
    camarilla_r3 = typical_price + (hl_range * 1.1 / 4.0)
    camarilla_s3 = typical_price - (hl_range * 1.1 / 4.0)
    camarilla_r4 = typical_price + (hl_range * 1.1 / 2.0)
    camarilla_s4 = typical_price - (hl_range * 1.1 / 2.0)
    
    # 12h EMA20 for trend filter
    ema_12h_20 = calculate_ema(h_close, 20)
    
    # Align all HTF arrays to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    ema_12h_20_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_20)
    
    # === 6h Indicators ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema_12h_20_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- 12h Trend Filter ---
        trend_bullish = close[i] > ema_12h_20_aligned[i]
        trend_bearish = close[i] < ema_12h_20_aligned[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Price Levels ---
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        
        # --- Position Management (Exit Logic) ---
        if in_position:
            # Exit conditions: opposite signal or mean reversion to pivot
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~12h)
            if min_hold:
                if position_side > 0:  # Long position
                    # Exit: price reaches R3 (take profit) or breaks below S3 (stop)
                    if close[i] >= r3 or close[i] < s3:
                        signals[i] = 0.0
                        in_position = False
                        position_side = 0
                    else:
                        signals[i] = SIZE
                else:  # Short position
                    # Exit: price reaches S3 (take profit) or breaks above R3 (stop)
                    if close[i] <= s3 or close[i] > r3:
                        signals[i] = 0.0
                        in_position = False
                        position_side = 0
                    else:
                        signals[i] = -SIZE
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long fade at S3: price rejects support in uptrend
        if close[i] <= s3 and trend_bullish and vol_ok:
            # Additional confirmation: price must show rejection (close > open)
            if close[i] > open[i] if 'open' in prices.columns else True:
                in_position = True
                position_side = 1
                entry_bar = i
                signals[i] = SIZE
        # Short fade at R3: price rejects resistance in downtrend
        elif close[i] >= r3 and trend_bearish and vol_ok:
            # Additional confirmation: price must show rejection (close < open)
            elif close[i] < open[i] if 'open' in prices.columns else True:
                in_position = True
                position_side = -1
                entry_bar = i
                signals[i] = -SIZE
        # Long breakout above R4: strong bullish continuation
        elif close[i] >= r4 and trend_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            signals[i] = SIZE
        # Short breakout below S4: strong bearish continuation
        elif close[i] <= s4 and trend_bearish and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals