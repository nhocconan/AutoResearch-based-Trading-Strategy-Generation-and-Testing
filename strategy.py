#!/usr/bin/env python3
"""
Experiment #199: 6h Camarilla Pivot + Volume Spike + Trend Filter

HYPOTHESIS: Camarilla pivot levels (calculated from 1d OHLC) act as significant support/resistance.
At 6h timeframe: 
- Long when price breaks above R4 with volume confirmation (>1.5x avg volume) and 12h trend up (close > EMA20)
- Short when price breaks below S4 with volume confirmation and 12h trend down (close < EMA20)
- Fade trades at R3/S3 with volume confirmation and opposite 12h trend (mean reversion in ranging markets)
- ATR-based stoploss (2.5x ATR) and take profit at pivot levels
This structure works in both bull/bear markets by combining breakout and mean-reversion logic based on higher timeframe trend.
Target: 75-175 total trades over 4 years (19-44/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_199_6h_camarilla_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA(20) on 12h close
    if len(df_12h) >= 20:
        close_12h = df_12h['close'].values
        ema_20_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
        ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    else:
        ema_20_12h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each 1d bar: based on previous day's OHLC
    if len(df_1d) >= 2:
        # Use previous day's OHLC to avoid look-ahead
        prev_close = df_1d['close'].shift(1).values
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        
        # Camarilla formulas
        range_ = prev_high - prev_low
        camarilla_h5 = prev_close + range_ * 1.1 / 2  # R4
        camarilla_h4 = prev_close + range_ * 1.1 / 4  # R3
        camarilla_h3 = prev_close + range_ * 1.1 / 6  # R2
        camarilla_l3 = prev_close - range_ * 1.1 / 6  # S2
        camarilla_l4 = prev_close - range_ * 1.1 / 4  # S3
        camarilla_l5 = prev_close - range_ * 1.1 / 2  # S4
        
        # Align to 6h timeframe (shifted by 1 for completed 1d bar)
        h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
        h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
        l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    else:
        h5_aligned = h4_aligned = h3_aligned = np.full(n, np.nan)
        l3_aligned = l4_aligned = l5_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF EMA20 and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(h5_aligned[i]) or np.isnan(l5_aligned[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: 12h EMA20 ---
        trend_up = close[i] > ema_20_12h_aligned[i]
        trend_down = close[i] < ema_20_12h_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Camarilla Levels ---
        r4 = h5_aligned[i]  # R4
        r3 = h4_aligned[i]  # R3
        s3 = l4_aligned[i]  # S3
        s4 = l5_aligned[i]  # S4
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at R3/S3 (fade level) or opposite pivot
                if close[i] >= r3:  # Take profit at R3 for longs
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at S3/R3 (fade level) or opposite pivot
                if close[i] <= s3:  # Take profit at S3 for shorts
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Breakout Long: Price > R4 + volume spike + 12h trend up
        breakout_long = (close[i] > r4) and volume_spike and trend_up
        
        # Breakout Short: Price < S4 + volume spike + 12h trend down
        breakout_short = (close[i] < s4) and volume_spike and trend_down
        
        # Fade Long: Price < S3 + volume spike + 12h trend down (mean reversion in downtrend)
        fade_long = (close[i] < s3) and volume_spike and trend_down
        
        # Fade Short: Price > R3 + volume spike + 12h trend up (mean reversion in uptrend)
        fade_short = (close[i] > r3) and volume_spike and trend_up
        
        if breakout_long or fade_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif breakout_short or fade_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals