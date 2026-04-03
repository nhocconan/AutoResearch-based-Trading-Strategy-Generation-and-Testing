#!/usr/bin/env python3
"""
Experiment #071: 6h Volume-Weighted RSI with 1d Trend Filter and ATR Stop

HYPOTHESIS: Volume-weighted RSI(14) on 6h timeframe, combined with 1d trend filter (price > EMA50 for long, < EMA50 for short) 
and ATR-based stoploss, creates a robust momentum strategy. Volume weighting prevents false signals during low-participation 
moves, while the 1d trend filter ensures alignment with higher timeframe direction. Targets 15-25 trades/year on 6h 
timeframe (60-100 total over 4 years) to minimize fee drag while capturing high-probability momentum shifts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_vwrsi_trend_stop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # Calculate Volume-Weighted RSI(14)
    vwrsi = np.full(n, np.nan)
    
    # Need at least 15 periods for RSI calculation (14 for gains/losses + 1 for initial)
    if n >= 15:
        # Calculate price changes
        delta = np.diff(close, prepend=close[0])
        
        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        
        # Volume-weight the gains and losses
        vol_gains = gains * volume
        vol_losses = losses * volume
        
        # Calculate average gains and losses using Wilder's smoothing (EWM with alpha=1/period)
        avg_vol_gain = pd.Series(vol_gains).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
        avg_vol_loss = pd.Series(vol_losses).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
        
        # Calculate RS and RSI
        rs = np.divide(avg_vol_gain, avg_vol_loss, out=np.full_like(avg_vol_gain, np.nan), where=avg_vol_loss!=0)
        vwrsi = 100 - (100 / (1 + rs))
        
        # For first 14 values, RSI is not defined
        vwrsi[:14] = np.nan
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(vwrsi[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in alignment with 1d trend ---
        price_above_1d_ema = close[i] > ema_50_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_50_1d_aligned[i]
        
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
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: VWRSI crosses above 30 (oversold recovery) in uptrend
        # Short: VWRSI crosses below 70 (overbought rejection) in downtrend
        
        # Need previous VWRSI value for crossover detection
        if i > 0 and not np.isnan(vwrsi[i-1]):
            long_condition = (
                vwrsi[i-1] <= 30 and vwrsi[i] > 30 and price_above_1d_ema  # Bullish crossover from oversold
            )
            
            short_condition = (
                vwrsi[i-1] >= 70 and vwrsi[i] < 70 and price_below_1d_ema  # Bearish crossover from overbought
            )
        else:
            long_condition = False
            short_condition = False
        
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