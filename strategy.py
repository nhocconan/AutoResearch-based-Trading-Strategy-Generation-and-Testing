#!/usr/bin/env python3
"""
Experiment #379: 6h Camarilla Pivot + 12h Trend Filter + Volume Spike

HYPOTHESIS: Camarilla pivot levels on 6h timeframe (S3/R3 for mean reversion, R4/S4 for breakout),
combined with 12h trend filter (price > EMA50) and 12h volume spike confirmation (> 1.8x average),
creates a robust strategy for 6h timeframe that works in both bull and bear markets.
Uses higher timeframes (12h) for signal direction and regime filtering, while 6h provides
entry timing at key Camarilla levels. Targets 12-37 trades/year (50-150 total over 4 years)
to minimize fee drag while capturing high-probability mean reversions and breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for trend filter and volume spike (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA(50) on 12h close
    if len(df_12h) >= 50:
        close_12h = df_12h['close'].values
        ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    else:
        ema_50_12h_aligned = np.full(n, np.nan)
    
    # Calculate volume ratio (current vs 20-period average) on 12h
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === 6h Indicators: Calculate Camarilla pivot levels from prior 12h bar ===
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)  # For breakout continuation
    camarilla_r4 = np.full(n, np.nan)
    
    # Pre-compute prior 12h OHLC for each 6h bar
    for i in range(n):
        current_time = prices.iloc[i]['open_time']
        # Find the most recent completed 12h bar before current 6h bar
        prior_12h_bars = df_12h[df_12h['open_time'] < current_time]
        if len(prior_12h_bars) > 0:
            prev_bar = prior_12h_bars.iloc[-1]
            ph = prev_bar['high']
            pl = prev_bar['low']
            pc = prev_bar['close']
            
            # Camarilla formulas
            range_ = ph - pl
            camarilla_s3[i] = pc - range_ * 1.1 / 4
            camarilla_r3[i] = pc + range_ * 1.1 / 4
            camarilla_s4[i] = pc + range_ * 1.1 / 2
            camarilla_r4[i] = pc - range_ * 1.1 / 2
        else:
            camarilla_s3[i] = np.nan
            camarilla_r3[i] = np.nan
            camarilla_s4[i] = np.nan
            camarilla_r4[i] = np.nan
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # Pre-compute before loop
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(camarilla_s3[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in trending markets ---
        price_above_12h_ema = close[i] > ema_50_12h_aligned[i]
        price_below_12h_ema = close[i] < ema_50_12h_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio_12h_aligned[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Camarilla R3 (strong resistance) or S3 (strong support)
                if close[i] >= camarilla_r3[i] or close[i] <= camarilla_s3[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Camarilla R3 (strong resistance) or S3 (strong support)
                if close[i] >= camarilla_r3[i] or close[i] <= camarilla_s3[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price at S3 (mean reversion) OR break above R4 with volume
        long_condition = (
            (close[i] <= camarilla_s3[i] * 1.001 and price_above_12h_ema) or  # S3 mean reversion in uptrend
            (close[i] > camarilla_r4[i] and volume_spike and price_above_12h_ema)  # Breakout with volume
        )
        
        # Short: Price at R3 (mean reversion) OR break below S4 with volume
        short_condition = (
            (close[i] >= camarilla_r3[i] * 0.999 and price_below_12h_ema) or  # R3 mean reversion in downtrend
            (close[i] < camarilla_s4[i] and volume_spike and price_below_12h_ema)  # Breakdown with volume
        )
        
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