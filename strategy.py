#!/usr/bin/env python3
"""
Experiment #034: 1h RSI(2) mean reversion with 4h trend filter and volume confirmation
HYPOTHESIS: In 1h timeframe, extreme RSI(2) readings (<10 for long, >90 for short) 
combined with 4h EMA(50) trend alignment and volume spikes capture high-probability 
mean reversion trades. Works in both bull/bear markets by following 4h trend direction 
while using 1h for precise timing. Target: 60-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_034_1h_rsi2_4h_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for EMA trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate EMA(50) on 4h close
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1h Indicators: RSI(2) for mean reversion ===
    def calculate_rsi(arr, period):
        delta = np.diff(arr, prepend=arr[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_2 = calculate_rsi(close, 2)
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # Already datetime64[ms], .hour works
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Warmup for RSI and EMA stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi_2[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            signals[i] = 0.0
            continue
        
        # --- 4h Trend Filter: EMA(50) direction ---
        price = close[i]
        ema_trend_up = price > ema_4h_aligned[i]   # Above 4h EMA = uptrend bias
        ema_trend_down = price < ema_4h_aligned[i] # Below 4h EMA = downtrend bias
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- RSI(2) Extreme Conditions ---
        rsi_oversold = rsi_2[i] < 10   # Extreme oversold
        rsi_overbought = rsi_2[i] > 90 # Extreme overbought
        
        # --- Exit Logic: RSI mean reversion ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions: RSI returns to neutral zone (40-60)
            if position_side > 0:  # Long position
                if rsi_2[i] >= 40:  # Exit long when RSI >= 40
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                if rsi_2[i] <= 60:  # Exit short when RSI <= 60
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Only trade when RSI extreme aligns with 4h trend AND volume spike
        if ema_trend_up:
            # Long: RSI oversold AND volume spike AND price above 4h EMA
            if rsi_oversold and volume_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        elif ema_trend_down:
            # Short: RSI overbought AND volume spike AND price below 4h EMA
            if rsi_overbought and volume_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # Exactly at 4h EMA (rare), do not trade
            signals[i] = 0.0
    
    return signals